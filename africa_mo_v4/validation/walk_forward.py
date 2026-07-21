"""validation/walk_forward.py — Walk-forward cross-validation with shape-safe prediction."""
import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler

import config.pipeline  as cfg_pipe
import config.variables as var
import config.paths     as paths
from preprocessing.imputer import PanelMICEImputer
from features.engineer     import FoldFeatureEngineer


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
                 save_model: bool = True) -> list:

        years   = sorted(df_raw["year"].unique())
        splits  = self.split(years)
        results = []
        best_model      = None
        best_scaler     = None   # CORRECTION: persist alongside the model (see utils/model_io.py)
        best_feat_cols  = None

        for fold_idx, (train_yr, test_yr) in enumerate(splits, 1):
            df_tr = df_raw[df_raw["year"].isin(train_yr)].copy()
            df_te = df_raw[df_raw["year"].isin(test_yr)].copy()

            # Step 1: Imputation fitted on train only
            imputer = PanelMICEImputer(max_iter=20, random_state=42)
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

            # Step 5: Train and predict
            try:
                model  = trainer_fn(X_tr2, y_tr2, X_val_s, y_val)
                y_pred = np.asarray(model.predict(X_te_s), dtype=float)
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
                **m,
            ))

            rmse_s = f"{m['RMSE']:.4f}" if not np.isnan(m['RMSE']) else "nan"
            r2_s   = f"{m['R2']:.4f}"   if not np.isnan(m['R2'])   else "nan"
            print(f"      Fold {fold_idx}/{len(splits)} — RMSE={rmse_s}  R²={r2_s}")

        # Save best model
        # CORRECTION (root-cause diagnostic report, Secções 6/7/8, recomendação #3):
        # persist the scaler and feat_cols alongside the model, instead of the
        # bare model object, so any downstream code that reloads this .pkl can
        # scale its inputs exactly as they were scaled during training.
        if save_model and best_model is not None:
            os.makedirs(paths.MODELS_DIR, exist_ok=True)
            pkl_path = os.path.join(
                paths.MODELS_DIR, f"modelo_{spec}_{model_name}.pkl"
            )
            from utils.model_io import save_model_bundle
            save_model_bundle(pkl_path, best_model, best_scaler, best_feat_cols)

        return results
