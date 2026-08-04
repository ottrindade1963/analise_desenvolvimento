"""validation/walk_forward.py — Walk-forward cross-validation with shape-safe prediction."""
import os
import pickle
import time
import resource
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler

# Computational-cost measurement uses resource.getrusage()'s peak RSS rather
# than tracemalloc. Two reasons: (1) tracemalloc only sees Python-level
# allocations — it misses the native/C buffers sklearn, xgboost, tensorflow
# and pymc actually allocate for the heavy lifting, so it would understate
# the real cost of exactly the models we care most about measuring; (2) its
# global start/stop tracing state is not safe to use concurrently across the
# worker threads evaluate_multi_seed() spawns per seed, whereas
# getrusage(RUSAGE_SELF) is a plain, thread-safe OS read. The trade-off made
# explicit: ru_maxrss is a monotonic high-water mark since process start, not
# an isolated per-fold delta — documented wherever it's reported.
def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB → MB (Linux)

import config.pipeline     as cfg_pipe
import config.variables    as var
import config.paths        as paths
import config.model_params as mp
from preprocessing.imputer import PanelMICEImputer
from features.engineer     import FoldFeatureEngineer

# Governance-under-test columns are named consistently by FoldFeatureEngineer:
# the PCA factor and its derivatives ("wgi_pca1_lag1", "wgi_pca1_ma3", ...),
# the raw WGI lags ("wgi_controle_corrupcao_lag1", ...), and the interaction
# terms ("inter_pca1_ied", "inter_wgicomp_comercio", ...). This name-based
# rule works for every spec in config/features.py::ABLATION_SPECS without
# needing to compare column sets across specs.
#
# Returns (target_lag_idx, governance_idx) — two tiers, not one flat list.
# CORRECTION (found while sandbox-testing the first version of this fix,
# which force-included every governance column uncapped): that backfires for
# specs with MANY governance columns (A3/A5 can have 12-16 raw-WGI-lag +
# interaction columns). With SARIMAX's max_exog=8 / Bayesian's
# max_features=10, uncapped inclusion fully consumes the budget and crowds
# out the target's OWN autoregressive lags — by far the strongest predictor
# in a slow-moving panel series — which measurably wrecked RMSE in testing
# (SARIMAX RMSE went from ~0.9 on A1_WDI_only to ~9.3 on A5_WDI_6WGI_inter on
# synthetic data, purely from displacing target_lag1/target_lag2, not
# because governance features are actually harmful).
#
# SARIMAXModel.fit()/BayesianModel.fit() receive this tuple and do their own
# capping against their own budget (max_exog / max_features respectively):
# tier 1 (target lags) is always kept, uncapped (only
# len(config.features.LAGS_TARGET) of them, so no crowd-out risk by itself);
# tier 2 (governance) is force-included but capped at half of whatever
# budget remains after tier 1 — enough for a non-trivial, meaningful
# governance presence (fixing the original blindness bug) without evicting
# every legacy predictor. The rest of the budget is still filled by
# correlation/variance as before.
def _governance_priority_idx(feat_cols: list):
    target_lag_idx = [i for i, c in enumerate(feat_cols)
                       if c.startswith(f"{var.TARGET}_lag")]
    governance_idx = [i for i, c in enumerate(feat_cols)
                       if c.startswith("wgi_") or c.startswith("inter_")]
    return target_lag_idx, governance_idx


@dataclass
class FoldResult:
    fold:        int
    spec:        str
    model:       str
    n_train:     int
    n_test:      int
    train_years: list
    test_years:  list
    RMSE:  float = np.nan
    MAE:   float = np.nan
    R2:    float = np.nan
    MAPE:  float = np.nan
    MASE:  float = np.nan
    # CORRECTION (multi-seed robustness + computational-cost requirements):
    seed:          int   = None
    fit_time_s:    float = np.nan
    predict_time_s: float = np.nan
    peak_mem_mb:   float = np.nan
    n_dropped_missing_target: int = 0


def _metrics(y_true, y_pred, y_train=None) -> dict:
    yt = np.asarray(y_true,  float)
    yp = np.asarray(y_pred, float)

    # FIX: align lengths — LSTM returns fewer rows due to lookback
    min_len = min(len(yt), len(yp))
    yt = yt[-min_len:]
    yp = yp[-min_len:]

    m  = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    n = len(yt)
    if n < 3:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, MAPE=np.nan, MASE=np.nan)

    res    = yt - yp
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-10
    mae    = float(np.mean(np.abs(res)))
    mape   = float(np.mean(np.abs(res / (np.abs(yt) + 1e-8))) * 100)

    if y_train is not None and len(y_train) > 1:
        naive_mae = float(np.mean(np.abs(np.diff(np.asarray(y_train, float)))))
    else:
        naive_mae = float(np.mean(np.abs(np.diff(yt)))) if n > 1 else 1e-10

    return dict(
        RMSE=float(np.sqrt(np.mean(res ** 2))),
        MAE=mae,
        R2=float(1 - ss_res / ss_tot),
        MAPE=mape,
        MASE=mae / (naive_mae + 1e-10),
    )


class WalkForwardCV:
    def __init__(self,
                 n_folds: int         = cfg_pipe.WF_N_FOLDS,
                 min_train_frac: float = cfg_pipe.WF_MIN_TRAIN):
        self.n_folds        = n_folds
        self.min_train_frac = min_train_frac

    def split(self, years: list) -> list:
        n       = len(years)
        min_tr  = max(5, int(n * self.min_train_frac))
        avail   = n - min_tr
        n_folds = min(self.n_folds, max(1, avail))
        fold_sz = max(1, avail // n_folds)
        splits  = []
        for f in range(n_folds):
            te_start = min_tr + f * fold_sz
            te_end   = min(n, te_start + fold_sz)
            if te_start >= n:
                break
            splits.append((years[:te_start], years[te_start:te_end]))
        return splits

    def evaluate(self, df_raw: pd.DataFrame, spec: str,
                 trainer_fn, model_name: str,
                 save_model: bool = True,
                 seed: int = None,
                 model_suffix: str = "") -> list:
        """
        Parameters
        ----------
        seed : int, optional
            Random seed threaded into the imputer and into trainer_fn (which
            forwards it to the underlying model's own random_state /
            random_seed). Defaults to config.model_params.SEED (42) for
            backward compatibility with single-seed callers.
        model_suffix : str, optional
            Appended to the saved .pkl filename (e.g. "_seed7"), so that
            evaluate_multi_seed() can persist one model per seed without
            overwriting the primary-seed artifact that
            explainability/ablation.py and pipeline.py expect to find at
            "modelo_{spec}_{model_name}.pkl".
        """
        seed = mp.SEED if seed is None else seed

        years   = sorted(df_raw["year"].unique())
        splits  = self.split(years)
        results = []
        best_model      = None
        best_scaler     = None   # CORRECTION: persist alongside the model (see utils/model_io.py)
        best_feat_cols  = None

        for fold_idx, (train_yr, test_yr) in enumerate(splits, 1):
            df_tr = df_raw[df_raw["year"].isin(train_yr)].copy()
            df_te = df_raw[df_raw["year"].isin(test_yr)].copy()

            # Step 1: Imputation fitted on train only.
            # CORRECTION (target-imputation leak — see preprocessing/imputer.py):
            # exclude_cols=[var.TARGET] stops MICE from fabricating missing
            # target values out of the very covariates used as model
            # features. random_state=seed so the multi-seed comparison also
            # covers the (usually minor) sensitivity of MICE itself to seed.
            imputer = PanelMICEImputer(max_iter=20, random_state=seed,
                                       exclude_cols=[var.TARGET])
            imputer.fit(df_tr)
            df_tr_imp = imputer.transform(df_tr)
            df_te_imp = imputer.transform(df_te)

            # Step 2: Feature engineering fitted on train only
            fe = FoldFeatureEngineer(spec=spec)
            fe.fit(df_tr_imp)
            df_combined    = pd.concat([df_tr_imp, df_te_imp], ignore_index=True)
            df_combined_fe = fe.transform(df_combined)

            df_tr_fe = df_combined_fe[df_combined_fe["year"].isin(train_yr)]
            df_te_fe = df_combined_fe[df_combined_fe["year"].isin(test_yr)]

            feat_cols = [
                c for c in df_combined_fe.select_dtypes(include=[np.number]).columns
                if c not in {"year", var.TARGET} and "country" not in c.lower()
            ]
            if not feat_cols or var.TARGET not in df_tr_fe.columns:
                continue

            X_tr = df_tr_fe[feat_cols].fillna(0).values
            y_tr = df_tr_fe[var.TARGET].values
            X_te = df_te_fe[feat_cols].fillna(0).values
            y_te = df_te_fe[var.TARGET].values

            # CORRECTION (target-imputation leak, cont.): the target is no
            # longer imputed, so rows whose true label was missing now show
            # up as real NaNs here instead of fabricated numbers. Test rows
            # were already safe (metrics() masks NaN), but sklearn/statsmodels
            # models cannot fit against a NaN label, so train rows must be
            # dropped explicitly. Test rows are dropped too for an honest,
            # consistent n_test (no fabricated ground truth in the eval set).
            n_dropped = 0
            mask_tr_y = ~pd.isna(y_tr)
            if not mask_tr_y.all():
                n_dropped += int((~mask_tr_y).sum())
                X_tr, y_tr = X_tr[mask_tr_y], y_tr[mask_tr_y]
            mask_te_y = ~pd.isna(y_te)
            if not mask_te_y.all():
                n_dropped += int((~mask_te_y).sum())
                X_te, y_te = X_te[mask_te_y], y_te[mask_te_y]

            if len(X_tr) < 5 or len(X_te) == 0:
                continue

            # Governance-under-test columns (see _governance_priority_idx) —
            # forced into SARIMAX's/Bayesian's internal top-K feature
            # reduction so those models are not structurally blind to the
            # very variables the ablation study is testing.
            priority_idx = _governance_priority_idx(feat_cols)

            # Step 3: Scaling fitted on train only
            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_te_s  = scaler.transform(X_te)

            # Step 4: Inner validation split
            n_val   = max(1, int(len(X_tr_s) * 0.15))
            X_val_s = X_tr_s[-n_val:]
            y_val   = y_tr[-n_val:]
            X_tr2   = X_tr_s[:-n_val]
            y_tr2   = y_tr[:-n_val]

            # Step 5: Train and predict, with computational-cost instrumentation
            fit_time_s = predict_time_s = np.nan
            peak_mem_mb = np.nan
            try:
                t0 = time.perf_counter()
                model = trainer_fn(X_tr2, y_tr2, X_val_s, y_val,
                                   priority_idx=priority_idx, seed=seed)
                fit_time_s = time.perf_counter() - t0
                peak_mem_mb = _peak_rss_mb()

                t1 = time.perf_counter()
                y_pred = np.asarray(model.predict(X_te_s), dtype=float)
                predict_time_s = time.perf_counter() - t1

                # FIX: length alignment handled inside _metrics
                m      = _metrics(y_te, y_pred, y_tr)
                best_model     = model
                best_scaler    = scaler
                best_feat_cols = feat_cols
            except Exception as exc:
                print(f"      Fold {fold_idx} [{model_name}] failed: {exc}")
                m = dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, MAPE=np.nan, MASE=np.nan)

            results.append(FoldResult(
                fold=fold_idx, spec=spec, model=model_name,
                n_train=len(X_tr2), n_test=len(X_te),
                train_years=list(train_yr), test_years=list(test_yr),
                seed=seed, fit_time_s=fit_time_s, predict_time_s=predict_time_s,
                peak_mem_mb=peak_mem_mb, n_dropped_missing_target=n_dropped,
                **m,
            ))

            # CORRECTION (user request — 4 decimal places are noise for RMSE/R2
            # at this scale): reduced from .4f to .3f.
            rmse_s = f"{m['RMSE']:.3f}" if not np.isnan(m['RMSE']) else "nan"
            r2_s   = f"{m['R2']:.3f}"   if not np.isnan(m['R2'])   else "nan"
            print(f"      Fold {fold_idx}/{len(splits)} — RMSE={rmse_s}  R²={r2_s}"
                  f"  seed={seed}"
                  + (f"  [{n_dropped} rows dropped: missing target]" if n_dropped else ""))

        # Save best model
        # CORRECTION (root-cause diagnostic report, Secções 6/7/8, recomendação #3):
        # persist the scaler and feat_cols alongside the model, instead of the
        # bare model object, so any downstream code that reloads this .pkl can
        # scale its inputs exactly as they were scaled during training.
        if save_model and best_model is not None:
            os.makedirs(paths.MODELS_DIR, exist_ok=True)
            pkl_path = os.path.join(
                paths.MODELS_DIR, f"modelo_{spec}_{model_name}{model_suffix}.pkl"
            )
            from utils.model_io import save_model_bundle
            save_model_bundle(pkl_path, best_model, best_scaler, best_feat_cols)

        return results

    def evaluate_multi_seed(self, df_raw: pd.DataFrame, spec: str,
                             trainer_fn, model_name: str,
                             seeds: list = None,
                             save_model: bool = True) -> list:
        """
        Run evaluate() once per seed (default: config.model_params.SEEDS) and
        return the concatenation of all FoldResult lists, each tagged with
        its seed. The FIRST seed in the list is treated as the primary one
        and keeps the unsuffixed model filename (so ablation.py / pipeline.py
        keep working against a single, well-defined artifact per spec×model);
        subsequent seeds are saved alongside with a "_seed{N}" suffix purely
        for the robustness comparison this function exists to produce.

        Uses joblib to parallelize across seeds (see docs/tooling_rationale
        for why joblib was chosen over pyspark/Dask for this workload size).
        """
        seeds = list(mp.SEEDS) if seeds is None else list(seeds)
        try:
            from joblib import Parallel, delayed
            n_jobs = min(len(seeds), max(1, (os.cpu_count() or 2) - 1))
        except ImportError:
            Parallel = None

        def _run(i, s):
            suffix = "" if i == 0 else f"_seed{s}"
            print(f"    ── seed {s} ({'primary' if i == 0 else 'secondary'}) ──")
            return self.evaluate(df_raw, spec, trainer_fn, model_name,
                                 save_model=save_model, seed=s, model_suffix=suffix)

        if Parallel is not None and len(seeds) > 1:
            # prefer="threads": trainer_fn is frequently a closure/lambda
            # (see MODEL_TRAINERS in pipeline.py / the notebook), which the
            # process-based "loky" backend cannot always pickle. Threading
            # avoids that entirely; the numeric libraries involved here
            # (numpy/sklearn/xgboost/tensorflow/pymc) release the GIL during
            # their actual fit/predict work, so this still parallelizes the
            # part that matters.
            nested = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_run)(i, s) for i, s in enumerate(seeds)
            )
        else:
            nested = [_run(i, s) for i, s in enumerate(seeds)]

        return [r for sub in nested for r in sub]
