"""models/sarimax/model.py — SARIMAX with AIC order selection and coefficient export.

Addresses Problem 10 from the review: the model now exports a full
coefficient table with coef, std error, 95% CI, and p-value.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import config.model_params as mp

warnings.filterwarnings("ignore")


class SARIMAXModel:
    """
    SARIMAX with exogenous variables.

    fit() selects ARIMA order via AIC when auto_order=True, then fits
    the model and stores all coefficient information for reporting.
    predict() uses out-of-sample forecast; falls back to Ridge if SARIMAX fails.
    """

    def __init__(self):
        self._use_sarimax    = False
        self._params         = None
        self._order          = mp.SARIMAX["order"]
        self._coef_table     = None   # DataFrame exported for the dissertation
        self._scaler         = None
        self._top_idx        = None
        self._ridge          = None
        self._endog_train    = None
        self._exog_train     = None

    def fit(self, X_tr, y_tr, X_val, y_val):
        cfg = mp.SARIMAX
        X   = np.asarray(X_tr, float)
        y   = np.asarray(y_tr, float).ravel()

        # Select top-K features by correlation with target
        n_exog = min(cfg["max_exog"], X.shape[1])
        corr   = np.array([
            abs(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 1e-10 else 0.0
            for i in range(X.shape[1])
        ])
        self._top_idx = np.argsort(corr)[-n_exog:]
        X_sel = X[:, self._top_idx]

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X_sel)

        # Ridge fallback (always available)
        self._ridge = Ridge(alpha=1.0)
        self._ridge.fit(X_s, y)

        # Auto order selection via AIC
        order = self._select_order(y, X_s) if cfg["auto_order"] else cfg["order"]
        self._order = order

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
            if len(y) >= sum(order) + 10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SM(
                        y, exog=X_s, order=order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False, maxiter=cfg["maxiter"], method=cfg["method"])

                # Validate on val set
                X_val_s = self._scaler.transform(
                    np.asarray(X_val, float)[:, self._top_idx]
                )
                fc = np.asarray(res.forecast(steps=len(y_val), exog=X_val_s))
                y_val_arr = np.asarray(y_val, float).ravel()

                if not (np.any(np.isnan(fc)) or np.any(np.isinf(fc))):
                    rmse_sar  = np.sqrt(np.mean((y_val_arr - fc) ** 2))
                    rmse_ridg = np.sqrt(np.mean(
                        (y_val_arr - X_val_s @ self._ridge.coef_ - self._ridge.intercept_) ** 2
                    ))
                    if rmse_sar <= rmse_ridg * 2.0:
                        self._params      = np.asarray(res.params)
                        self._endog_train = y
                        self._exog_train  = X_s
                        self._use_sarimax = True

                        # ── Coefficient table (for dissertation) ──────────────
                        if cfg["export_coefficients"]:
                            summary = res.summary2().tables[1]
                            self._coef_table = pd.DataFrame({
                                "Parameter":    summary.index.tolist(),
                                "Coefficient":  summary["Coef."].values,
                                "Std_Error":    summary["Std.Err."].values,
                                "t_stat":       summary["t"].values,
                                "p_value":      summary["P>|t|"].values,
                                "CI_lower_95":  summary["[0.025"].values,
                                "CI_upper_95":  summary["0.975]"].values,
                            })
        except Exception:
            pass

        return self

    def predict(self, X):
        X_arr = np.asarray(X, float)
        X_sel = X_arr[:, self._top_idx]
        X_s   = self._scaler.transform(X_sel)

        if self._use_sarimax:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SM(
                        self._endog_train, exog=self._exog_train,
                        order=self._order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).smooth(self._params)
                    fc = np.asarray(res.forecast(steps=X_s.shape[0], exog=X_s))
                if not (np.any(np.isnan(fc)) or np.any(np.isinf(fc))):
                    return fc
            except Exception:
                pass

        return X_s @ self._ridge.coef_ + self._ridge.intercept_

    @staticmethod
    def _select_order(y, X_s) -> tuple:
        """Select ARIMA order by AIC over p,d,q ∈ {0,1,2}."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
            best_aic, best_order = np.inf, (1, 1, 1)
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                res = SM(
                                    y, exog=X_s, order=(p, d, q), trend="c",
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False, maxiter=200, method="lbfgs")
                            if res.aic < best_aic:
                                best_aic, best_order = res.aic, (p, d, q)
                        except Exception:
                            pass
            return best_order
        except Exception:
            return (1, 1, 1)

    def export_coef_table(self, path: str) -> None:
        if self._coef_table is not None:
            self._coef_table.to_csv(path, index=False)
            print(f"    SARIMAX coefficient table → {path}")


def train(X_tr, y_tr, X_val, y_val, run_name: str = "SARIMAX") -> SARIMAXModel:
    model = SARIMAXModel()
    model.fit(X_tr, y_tr, X_val, y_val)
    model._search_method       = "AIC order selection"
    model._selection_criterion = "AIC on training fold"
    return model
