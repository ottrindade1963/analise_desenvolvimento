"""pipeline.py — Main pipeline orchestrator.

Architecture (per the document recommendation):

    pipeline.py
        │
        ├── config/          paths · variables · features · model_params · pipeline
        ├── data/            extraction (WDI + WGI via World Bank API)
        ├── preprocessing/   PanelMICEImputer · PanelScaler  (sklearn Transformers)
        ├── features/        FoldFeatureEngineer              (sklearn Transformer)
        ├── validation/      WalkForwardCV
        ├── models/          rf · xgb · sarimax · lstm · bayesian
        ├── tuning/          Optuna TPE search for all tree models
        ├── explainability/  SHAP · Permutation · Ablation
        ├── reports/         auto-generated dissertation tables + Markdown summary
        ├── figures/         all plots
        └── utils/           MLflow tracking · metadata

Key methodological guarantees
──────────────────────────────
1. MICE imputation fitted ONLY on training data inside each fold   → no look-ahead
2. PCA fitted ONLY on training data inside each fold               → no look-ahead
3. StandardScaler fitted ONLY on training data inside each fold    → no leakage
4. Lag/rolling features computed with pandas shift/rolling         → backward-only
5. Optuna TPE search with inner validation split inside each fold  → correct CV
6. LSTM lookback = config.LSTM['lookback'] ≥ 3                    → true sequences
7. Full hyperparameter table exported                              → reproducibility
8. Five ablation specifications                                    → research hypothesis
9. Bayesian PPCs + R-hat + ESS exported                           → diagnostic
10. SARIMAX coefficient table (SE, CI95, p-val) exported          → interpretability
"""
import os
import sys
import time
import glob
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config.paths     as paths
import config.variables as var
import config.features  as feat
import config.pipeline  as cfg_pipe
import config.model_params as mp

from preprocessing.imputer    import PanelMICEImputer
from preprocessing.scaler     import PanelScaler
from features.engineer        import FoldFeatureEngineer
from validation.walk_forward  import WalkForwardCV
from tuning.optuna_search     import export_hyperparameter_table
from explainability.shap_analysis import shap_tree_analysis, shap_kernel_analysis
from explainability.permutation   import permutation_importance
from explainability.ablation      import run_ablation
from reports.report_generator     import run_all_reports
from utils.tracking               import log_metadata


# ── Model trainers ────────────────────────────────────────────────────────────
from models.rf.model       import train as train_rf
from models.xgb.model      import train as train_xgb
from models.sarimax.model  import train as train_sarimax
from models.lstm.model     import train as train_lstm
from models.bayesian.model import train as train_bayesian

MODEL_TRAINERS = {
    "RandomForest":         train_rf,
    "XGBoost":              train_xgb,
    "SARIMAX":              train_sarimax,
    "LSTM":                 train_lstm,
    "Bayes_Partial":        lambda X_tr,y_tr,X_va,y_va: train_bayesian(X_tr,y_tr,X_va,y_va,"partial"),
    "Bayes_Complete":       lambda X_tr,y_tr,X_va,y_va: train_bayesian(X_tr,y_tr,X_va,y_va,"complete"),
}


# ── Step helpers ──────────────────────────────────────────────────────────────

def _load_clean_data() -> pd.DataFrame:
    """Load the INNER JOIN of clean WDI + WGI."""
    path = os.path.join(paths.AGGREGATED_DIR, "agregado_inner_join.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Run data extraction first. Expected: {path}"
        )
    df = pd.read_csv(path)
    print(f"  Data loaded: {df.shape}  "
          f"({df['country_code'].nunique()} countries, "
          f"{df['year'].min()}–{df['year'].max()})")
    return df


def _build_spec_datasets(df_raw: pd.DataFrame) -> dict:
    """
    Build one feature-engineered dataset per ablation specification.
    PCA is fitted on the temporally first PCA_TRAIN_FRAC of each country.
    """
    datasets = {}
    for spec_name, spec_cfg in feat.ABLATION_SPECS.items():
        print(f"  Building features for spec: {spec_name}")
        fe = FoldFeatureEngineer(spec=spec_name)
        fe.fit(df_raw)                       # PCA fitted on 80% per country
        df_fe = fe.transform(df_raw)
        out_path = os.path.join(paths.FEATURES_DIR, f"{spec_name}_features.csv")
        df_fe.to_csv(out_path, index=False)
        datasets[spec_name] = df_fe
    return datasets


def _train_and_evaluate(
    df_raw: pd.DataFrame,
    spec_name: str,
    wf: WalkForwardCV,
) -> tuple[list, list]:
    """Train all models for one spec using walk-forward CV."""
    all_fold_results = []
    hp_records       = []

    for mod_name, trainer_fn in MODEL_TRAINERS.items():
        print(f"\n  [{spec_name}] [{mod_name}]")
        fold_results = wf.evaluate(df_raw, spec_name, trainer_fn, mod_name)
        all_fold_results.extend(fold_results)

        # ── Train final model on full train window → save ──────────────────
        years      = sorted(df_raw["year"].unique())
        split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
        train_yr   = years[:split_idx]
        test_yr    = years[split_idx:]

        df_tr = df_raw[df_raw["year"].isin(train_yr)].copy()
        df_te = df_raw[df_raw["year"].isin(test_yr)].copy()

        imputer = PanelMICEImputer()
        imputer.fit(df_tr)
        df_tr_imp = imputer.transform(df_tr)
        df_te_imp = imputer.transform(df_te)

        fe     = FoldFeatureEngineer(spec=spec_name)
        combined = pd.concat([df_tr_imp, df_te_imp], ignore_index=True)
        fe.fit(df_tr_imp)
        df_combined_fe = fe.transform(combined)

        df_tr_fe = df_combined_fe[df_combined_fe["year"].isin(train_yr)]
        df_te_fe = df_combined_fe[df_combined_fe["year"].isin(test_yr)]

        feat_cols = [
            c for c in df_combined_fe.select_dtypes(include=[np.number]).columns
            if c not in {"year", var.TARGET} and "country" not in c.lower()
        ]
        if not feat_cols:
            continue

        from sklearn.preprocessing import StandardScaler
        scaler    = StandardScaler()
        X_tr_s    = scaler.fit_transform(df_tr_fe[feat_cols].values)
        X_te_s    = scaler.transform(df_te_fe[feat_cols].values)
        y_tr      = df_tr_fe[var.TARGET].values
        y_te      = df_te_fe[var.TARGET].values

        n_val   = max(1, int(len(X_tr_s) * 0.15))
        X_va_s  = X_tr_s[-n_val:]
        y_va    = y_tr[-n_val:]
        X_tr2   = X_tr_s[:-n_val]
        y_tr2   = y_tr[:-n_val]

        try:
            final_model = trainer_fn(X_tr2, y_tr2, X_va_s, y_va)

            model_path = os.path.join(paths.MODELS_DIR,
                                      f"modelo_{spec_name}_{mod_name}.pkl")
            # CORRECTION (root-cause diagnostic report, Secções 6/7/8, recomendação #3):
            # persist scaler + feat_cols alongside the model (this is the file that
            # explainability/ablation.py, explainability/innovations.py, and
            # _run_explainability() below actually load — it overwrites the one
            # written inside wf.evaluate()).
            from utils.model_io import save_model_bundle
            save_model_bundle(model_path, final_model, scaler, feat_cols)

            # Export SARIMAX coefficient table
            if mod_name == "SARIMAX" and hasattr(final_model, "export_coef_table"):
                coef_path = os.path.join(paths.REPORTS_DIR,
                                         f"sarimax_{spec_name}_coef.csv")
                final_model.export_coef_table(coef_path)

            # Export Bayesian diagnostics
            if "Bayes" in mod_name and hasattr(final_model, "export_diagnostics"):
                diag_dir = os.path.join(paths.EXPLAINABILITY_DIR, "bayesian")
                final_model.export_diagnostics(diag_dir)

            hp_records.append({
                "Specification":      spec_name,
                "Model":              mod_name,
                "Search_Method":      getattr(final_model, "_search_method", "—"),
                "N_Trials":           mp.RF["n_trials"] if "Forest" in mod_name else "—",
                "Selection_Criterion":getattr(final_model, "_selection_criterion", "—"),
                "Seed":               getattr(final_model, "_seed", 42),
                "Best_Params":        str(getattr(final_model, "_best_params", "—")),
            })

        except Exception as exc:
            print(f"    Final model failed: {exc}")

    return all_fold_results, hp_records


# CORRECTION (root-cause diagnostic report, Secção 10.2 / recomendação #4):
# the old name "WDI_plus_PCA1" predates the current ABLATION_SPECS naming
# convention (A1_WDI_only, A2_WDI_PCA1, ...) and never matched any real key,
# so the KernelExplainer branch below never ran. The real equivalent spec is
# A2_WDI_PCA1 (WDI + single PCA governance factor) — made explicit here.
REFERENCE_SPEC_FOR_KERNEL_SHAP = "A2_WDI_PCA1"


def _run_explainability(spec_datasets: dict) -> None:
    """SHAP + permutation importance for all tree models on main datasets."""
    print("\n" + "=" * 60)
    print("  EXPLAINABILITY")
    print("=" * 60)
    from utils.model_io import load_model_bundle
    exp_dir = paths.EXPLAINABILITY_DIR
    tree_models = ["RandomForest", "XGBoost"]

    for spec_name, df in spec_datasets.items():
        if "Sintetico" in spec_name:
            continue

        feat_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in {"year", var.TARGET} and "country" not in c.lower()
        ]
        years     = sorted(df["year"].unique())
        split_idx = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
        test_yr   = years[split_idx:]

        for mod_name in tree_models:
            pkl = os.path.join(paths.MODELS_DIR, f"modelo_{spec_name}_{mod_name}.pkl")
            if not os.path.exists(pkl):
                continue
            model, scaler, trained_feat_cols = load_model_bundle(pkl)
            cols = trained_feat_cols if trained_feat_cols else feat_cols
            cols = [c for c in cols if c in df.columns]

            X_all_raw  = df[cols].fillna(0)
            X_test_raw = df[df["year"].isin(test_yr)][cols].fillna(0)
            y_test     = df[df["year"].isin(test_yr)][var.TARGET].values

            # CORRECTION (root-cause diagnostic report, Secção 8, achado nº7 /
            # recomendação #3): the model was trained on StandardScaler-scaled
            # data (see _train_and_evaluate above). X_all/X_test used to be
            # passed to SHAP/permutation in raw, unscaled units — apply the
            # persisted scaler here before calling either function.
            if scaler is not None:
                X_all  = pd.DataFrame(scaler.transform(X_all_raw.values),  columns=cols, index=X_all_raw.index)
                X_test = pd.DataFrame(scaler.transform(X_test_raw.values), columns=cols, index=X_test_raw.index)
            else:
                print(f"    [aviso] {spec_name}_{mod_name}: pickle sem scaler persistido "
                      f"(formato anterior à correcção) — SHAP/permutação usam dados brutos.")
                X_all, X_test = X_all_raw, X_test_raw

            label = f"{spec_name}_{mod_name}"
            print(f"  SHAP + Permutation → {label}")
            try:
                shap_tree_analysis(model, X_all, label, exp_dir)
            except Exception as exc:
                print(f"    SHAP failed: {exc}")
            try:
                permutation_importance(model, X_test, y_test, label, exp_dir)
            except Exception as exc:
                print(f"    Permutation failed: {exc}")

        # KernelExplainer for non-tree models on the reference spec
        if spec_name == REFERENCE_SPEC_FOR_KERNEL_SHAP:
            cols_ref  = [c for c in feat_cols if c in df.columns]
            X_all_ref = df[cols_ref].fillna(0)
            for mod_name in ["SARIMAX", "LSTM", "Bayes_Partial"]:
                pkl = os.path.join(paths.MODELS_DIR, f"modelo_{spec_name}_{mod_name}.pkl")
                if not os.path.exists(pkl):
                    continue
                model, scaler, trained_feat_cols = load_model_bundle(pkl)
                cols = [c for c in (trained_feat_cols or cols_ref) if c in df.columns]
                X_bg = df[cols].fillna(0)
                if scaler is not None:
                    X_bg = pd.DataFrame(scaler.transform(X_bg.values), columns=cols, index=X_bg.index)
                label = f"{spec_name}_{mod_name}"
                try:
                    shap_kernel_analysis(model, X_bg,
                                         X_bg.sample(min(200, len(X_bg)), random_state=42),
                                         label, exp_dir)
                except Exception as exc:
                    print(f"    KernelExplainer failed ({label}): {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "═" * 70)
    print("  AFRICA & MIDDLE EAST INDUSTRIAL ANALYSIS PIPELINE — v4.0")
    print("  Professional ML Pipeline | Walk-Forward CV | Optuna | MLflow")
    print("═" * 70)

    t_global = time.time()
    all_fold_results = []
    all_hp_records   = []

    # ── 1. Load clean aggregated data ─────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df_raw = _load_clean_data()

    # ── 2. Walk-forward CV for all specs and models ───────────────────────────
    print("\n[2/6] Walk-forward cross-validation + model training...")
    wf = WalkForwardCV()

    for spec_name in feat.ABLATION_SPECS:
        print(f"\n{'─'*60}")
        print(f"  Specification: {spec_name}")
        fold_res, hp_recs = _train_and_evaluate(df_raw, spec_name, wf)
        all_fold_results.extend(fold_res)
        all_hp_records.extend(hp_recs)

    # ── 3. Save walk-forward results ──────────────────────────────────────────
    print("\n[3/6] Saving results...")
    df_wf = pd.DataFrame([
        {"fold": r.fold, "spec": r.spec, "model": r.model,
         "RMSE": r.RMSE, "MAE": r.MAE, "R2": r.R2, "MASE": r.MASE,
         "n_train": r.n_train, "n_test": r.n_test}
        for r in all_fold_results
    ])
    wf_path = os.path.join(paths.REPORTS_DIR, "walkforward_results.csv")
    df_wf.to_csv(wf_path, index=False)

    hp_path = export_hyperparameter_table(all_hp_records)

    # ── 4. Explainability ─────────────────────────────────────────────────────
    print("\n[4/6] Explainability (SHAP + Permutation)...")
    spec_datasets = _build_spec_datasets(df_raw)
    _run_explainability(spec_datasets)

    # ── 5. Ablation study ─────────────────────────────────────────────────────
    print("\n[5/6] Ablation study...")
    years      = sorted(df_raw["year"].unique())
    split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
    test_cutoff = years[split_idx]
    abl_dir    = os.path.join(paths.EXPLAINABILITY_DIR, "ablation")

    run_ablation(
        spec_datasets=spec_datasets,
        model_names=list(MODEL_TRAINERS.keys()),
        model_dir=paths.MODELS_DIR,
        out_dir=abl_dir,
        test_year_cutoff=test_cutoff,
    )

    # ── 6. Reports ────────────────────────────────────────────────────────────
    print("\n[6/6] Generating dissertation reports...")

    # CORRECTION (root-cause diagnostic report, Secção 10.1 / recomendação #5):
    # df_wf has one row per (fold, spec, model). "best_RMSE" used to report only
    # the single minimum fold-level RMSE, with no label distinguishing it from
    # the mean-per-spec×model figure in table_performance.csv — the two are
    # different statistics of the same data and were easy to misread as
    # inconsistent. Both are now reported, explicitly labelled.
    best_rmse_single_fold = float(df_wf["RMSE"].dropna().min()) if not df_wf.empty else np.nan
    best_row_single_fold  = df_wf.loc[df_wf["RMSE"].idxmin()] if not df_wf.empty else {}

    if not df_wf.empty:
        df_mean_grp = df_wf.groupby(["spec", "model"])["RMSE"].mean().reset_index()
        best_row_mean = df_mean_grp.loc[df_mean_grp["RMSE"].idxmin()]
        best_rmse_mean = float(best_row_mean["RMSE"])
    else:
        best_row_mean  = {}
        best_rmse_mean = np.nan

    # CORRECTION (root-cause diagnostic report, Secção 10.2 / recomendação #4):
    # "sarimax_WDI_plus_PCA1_coef.csv" never matched any file actually written
    # by _train_and_evaluate() (real files are sarimax_{spec_name}_coef.csv).
    # Aggregate every per-spec SARIMAX coefficient file that was produced into
    # one combined table instead of pointing at a name that never existed.
    sarimax_coef_files = sorted(glob.glob(os.path.join(paths.REPORTS_DIR, "sarimax_*_coef.csv")))
    sarimax_coef_csv = None
    if sarimax_coef_files:
        frames = []
        for fp in sarimax_coef_files:
            spec_from_name = os.path.basename(fp)[len("sarimax_"):-len("_coef.csv")]
            d = pd.read_csv(fp)
            d.insert(0, "Specification", spec_from_name)
            frames.append(d)
        combined = pd.concat(frames, ignore_index=True)
        sarimax_coef_csv = os.path.join(paths.REPORTS_DIR, "sarimax_all_specs_coef.csv")
        combined.to_csv(sarimax_coef_csv, index=False)
        print(f"  [correction] SARIMAX coef table aggregated from {len(sarimax_coef_files)} "
              f"spec files → {sarimax_coef_csv}")

    run_all_reports(
        results_csv=wf_path,
        hp_csv=hp_path,
        sarimax_coef_csv=sarimax_coef_csv,
        ablation_csv=os.path.join(abl_dir, "ablation_results.csv"),
        ablation_dm_csv=os.path.join(abl_dir, "ablation_dm_tests.csv"),
        summary_kv={
            "best_RMSE_single_fold (mínimo entre as linhas de fold individuais)": best_rmse_single_fold,
            "best_model_single_fold":  getattr(best_row_single_fold, "model", "—"),
            "best_spec_single_fold":   getattr(best_row_single_fold, "spec",  "—"),
            "best_RMSE_mean_per_group (média por especificação×modelo — comparável à tabela de desempenho)": best_rmse_mean,
            "best_model_mean_per_group": best_row_mean.get("model", "—") if isinstance(best_row_mean, dict) else best_row_mean["model"],
            "best_spec_mean_per_group":  best_row_mean.get("spec",  "—") if isinstance(best_row_mean, dict) else best_row_mean["spec"],
            "n_models":   len(MODEL_TRAINERS),
            "n_specs":    len(feat.ABLATION_SPECS),
            "n_folds":    cfg_pipe.WF_N_FOLDS,
        },
    )
    best_rmse = best_rmse_single_fold

    # ── Metadata ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    log_metadata(
        step="pipeline_v4",
        params={
            "n_models": len(MODEL_TRAINERS),
            "n_specs":  len(feat.ABLATION_SPECS),
            "n_folds":  cfg_pipe.WF_N_FOLDS,
            "lookback": mp.LSTM["lookback"],
            "optuna_trials": mp.RF["n_trials"],
        },
        metrics={"total_time_s": round(elapsed, 1), "best_RMSE": best_rmse},
        output_files=[wf_path, hp_path],
    )

    print(f"\n{'═'*70}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'═'*70}")
    print(f"  Walk-forward results  → {wf_path}")
    print(f"  Hyperparameter table  → {hp_path}")
    print(f"  Reports               → {paths.REPORTS_DIR}/")
    print(f"  Figures               → {paths.EXPLAINABILITY_DIR}/")


if __name__ == "__main__":
    run_pipeline()
