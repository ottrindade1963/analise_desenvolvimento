"""reports/report_generator.py — Automatic dissertation-ready report generation."""
import os
import datetime
import numpy as np
import pandas as pd
import config.paths    as paths
import config.pipeline as cfg


def _to_latex(df: pd.DataFrame, caption: str, label: str,
              float_fmt: str = "%.3f") -> str:
    return df.to_latex(
        index=False, caption=caption, label=label,
        float_format=float_fmt, na_rep="—", escape=True,
    )


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from walk-forward CSV to report format."""
    df = df.rename(columns={
        "spec":    "Specification",
        "model":   "Model",
        "Dataset": "Specification",
    })
    return df


def generate_performance_table(results_csv: str) -> dict:
    if not os.path.exists(results_csv):
        return {}

    df = _normalise(pd.read_csv(results_csv))

    metric_cols = [c for c in ["RMSE","MAE","R2","MASE","MAPE"] if c in df.columns]
    id_cols     = [c for c in ["Specification","Model"] if c in df.columns]

    # Aggregate folds → mean per Specification × Model
    if id_cols and metric_cols:
        df = df.groupby(id_cols)[metric_cols].mean().reset_index()

    cols   = [c for c in id_cols + metric_cols if c in df.columns]
    df_tbl = df[cols].round(4)

    csv_path = os.path.join(paths.REPORTS_DIR, "table_performance.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_performance.tex")
    df_tbl.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df_tbl,
                          "Walk-forward CV — mean RMSE/MAE/R² per model and specification",
                          "tab:performance"))
    print(f"  [report] Performance table → {csv_path}  ({len(df_tbl)} rows)")
    return {"csv": csv_path, "tex": tex_path}


def generate_hyperparameter_table(hp_csv: str) -> dict:
    if not hp_csv or not os.path.exists(hp_csv):
        return {}
    df = pd.read_csv(hp_csv)
    if df.empty:
        return {}
    csv_path = os.path.join(paths.REPORTS_DIR, "table_hyperparameters.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_hyperparameters.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df, "Hyperparameter search: method, space, and selected values",
                          "tab:hyperparameters"))
    print(f"  [report] Hyperparameter table → {csv_path}")
    return {"csv": csv_path, "tex": tex_path}


def generate_sarimax_coef_table(coef_csv: str) -> dict:
    if not coef_csv or not os.path.exists(coef_csv):
        return {}
    df = pd.read_csv(coef_csv)
    df_fmt = df.copy()
    for col in ["Coefficient","Std_Error","CI_lower_95","CI_upper_95"]:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.4f}")
    if "p_value" in df_fmt.columns:
        df_fmt["p_value"] = df_fmt["p_value"].apply(
            lambda x: f"{x:.4f}{'***' if x<.01 else '**' if x<.05 else '*' if x<.1 else ''}"
        )
    csv_path = os.path.join(paths.REPORTS_DIR, "table_sarimax_coef.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_sarimax_coef.tex")
    df_fmt.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df_fmt,
                          "SARIMAX coefficient estimates (*** p<0.01; ** p<0.05; * p<0.10)",
                          "tab:sarimax_coef"))
    print(f"  [report] SARIMAX coefficient table → {csv_path}")
    return {"csv": csv_path, "tex": tex_path}


def generate_ablation_table(ablation_csv: str, dm_csv: str) -> dict:
    if not ablation_csv or not os.path.exists(ablation_csv):
        return {}
    df_abl = pd.read_csv(ablation_csv)
    cols   = ["Specification","Model","RMSE","R2"]
    df_out = df_abl[[c for c in cols if c in df_abl.columns]].copy()

    if dm_csv and os.path.exists(dm_csv):
        df_dm = pd.read_csv(dm_csv)
        # Handle both column name variants
        alt_col = "Alternative" if "Alternative" in df_dm.columns else "Specification"
        sig_col = next((c for c in ["Significant_5pct","Significant"] if c in df_dm.columns), None)
        merge_cols = [alt_col, "Model", "DM_p_value"] + ([sig_col] if sig_col else [])
        merge_cols = [c for c in merge_cols if c in df_dm.columns]
        dm_pv = df_dm[merge_cols].rename(columns={
            alt_col:  "Specification",
            "DM_p_value": "DM_p(vs baseline)",
        })
        df_out = df_out.merge(dm_pv, on=["Specification","Model"], how="left")

    csv_path = os.path.join(paths.REPORTS_DIR, "table_ablation.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_ablation.tex")
    df_out.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df_out,
                          "Ablation study: impact of governance specifications on RMSE",
                          "tab:ablation"))
    print(f"  [report] Ablation table → {csv_path}  ({len(df_out)} rows)")
    return {"csv": csv_path, "tex": tex_path}


def generate_executive_summary(results: dict) -> str:
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Pipeline Executive Summary",
        f"*Generated: {now}*\n",
        f"**Project**: {cfg.PROJECT_NAME} v{cfg.PROJECT_VERSION}\n",
        "## Method",
        "Walk-forward cross-validation (5 folds) with fold-level MICE imputation, "
        "StandardScaler, and PCA applied exclusively on training data. "
        "Optuna TPE hyperparameter search (50 trials per model). "
        "Ablation study over 5 governance specifications.\n",
        "## Key results",
    ]
    for key, val in results.items():
        fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"- **{key}**: {fmt}")

    md   = "\n".join(lines)
    path = os.path.join(paths.REPORTS_DIR, "executive_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  [report] Executive summary → {path}")
    return path


def run_all_reports(
    results_csv:      str | None = None,
    hp_csv:           str | None = None,
    sarimax_coef_csv: str | None = None,
    ablation_csv:     str | None = None,
    ablation_dm_csv:  str | None = None,
    summary_kv:       dict | None = None,
) -> list:
    print("\n" + "=" * 60)
    print("  REPORTS: generating dissertation tables")
    print("=" * 60)

    generated = []

    if results_csv:
        r = generate_performance_table(results_csv)
        generated.extend(r.values())

    if hp_csv:
        r = generate_hyperparameter_table(hp_csv)
        generated.extend(r.values())

    if sarimax_coef_csv:
        r = generate_sarimax_coef_table(sarimax_coef_csv)
        generated.extend(r.values())

    if ablation_csv:
        r = generate_ablation_table(ablation_csv, ablation_dm_csv or "")
        generated.extend(r.values())

    if summary_kv:
        p = generate_executive_summary(summary_kv)
        generated.append(p)

    print(f"\n  {len(generated)} report files generated in {paths.REPORTS_DIR}/")
    return generated
