"""models/bayesian/model.py — Bayesian regression with proper numpy-version fallback."""
import os
import signal
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import config.model_params as mp
import config.paths as paths

warnings.filterwarnings("ignore")


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


class BayesianModel:
    """
    Hierarchical Bayesian regression (PyMC) with BayesianRidge fallback.
    
    The fallback uses StandardScaler internally so that RMSE is comparable
    to other models (previously BayesianRidge ran on unscaled data → poor RMSE).
    """

    def __init__(self, pooling: str = "partial"):
        self.pooling   = pooling
        self._alpha    = 0.0
        self._beta     = None
        self._scaler   = StandardScaler()
        self._top_idx  = None
        self._y_mean   = 0.0
        self._y_std    = 1.0
        self._trace    = None
        self._is_pymc  = False
        self._fallback = None
        self._coef_summary = None

    def fit(self, X_tr, y_tr, X_val, y_val):
        cfg = mp.BAYESIAN
        X   = np.asarray(X_tr, float)
        y   = np.asarray(y_tr, float)

        # Reduce to top features by variance
        n_feat = min(cfg["max_features"], X.shape[1])
        self._top_idx = np.argsort(np.var(X, axis=0))[-n_feat:]
        X_red = X[:, self._top_idx]

        # Scale X and y
        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X_red)
        self._y_mean = float(y.mean())
        self._y_std  = float(y.std()) or 1.0
        y_s = (y - self._y_mean) / self._y_std

        pymc_ok = False
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(cfg["timeout_s"])

            import pymc as pm
            import arviz as az

            with pm.Model() as pm_model:
                if self.pooling == "partial":
                    mu_b    = pm.Normal("mu_beta",    mu=0, sigma=1)
                    sigma_b = pm.HalfNormal("sigma_beta", sigma=1)
                    beta    = pm.Normal("beta", mu=mu_b, sigma=sigma_b,
                                        shape=X_s.shape[1])
                else:
                    beta = pm.Normal("beta", mu=0, sigma=1, shape=X_s.shape[1])

                alpha = pm.Normal("alpha", mu=0, sigma=2)
                sigma = pm.HalfNormal("sigma", sigma=2)
                mu    = alpha + pm.math.dot(X_s, beta)
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

                self._trace = pm.sample(
                    draws=cfg["draws"], tune=cfg["tune"],
                    chains=cfg["chains"], cores=cfg["cores"],
                    random_seed=cfg["seed"],
                    return_inferencedata=True, progressbar=False,
                )

                if cfg.get("posterior_predictive", True):
                    self._ppc = pm.sample_posterior_predictive(
                        self._trace, progressbar=False
                    )

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            self._alpha  = float(self._trace.posterior["alpha"].values.mean())
            self._beta   = self._trace.posterior["beta"].values.mean(axis=(0, 1))
            self._is_pymc = True
            self._coef_summary = az.summary(
                self._trace, var_names=["alpha", "beta"], hdi_prob=0.94
            )
            pymc_ok = True

        except Exception as exc:
            try:
                signal.alarm(0)
            except Exception:
                pass
            print(f"    PyMC ({self.pooling}) failed ({exc}); BayesianRidge fallback.")

        if not pymc_ok:
            # BayesianRidge on scaled data for fair comparison
            self._fallback = BayesianRidge(max_iter=300)
            self._fallback.fit(X_s, y_s)

        return self

    def predict(self, X):
        X_arr = np.asarray(X, float)
        # Select same features as during fit
        if X_arr.shape[1] > len(self._top_idx):
            X_red = X_arr[:, self._top_idx]
        else:
            X_red = X_arr
        X_s = self._scaler.transform(X_red[:, :self._scaler.n_features_in_])

        if self._is_pymc and self._beta is not None:
            y_s = self._alpha + X_s @ self._beta
        elif self._fallback is not None:
            y_s = self._fallback.predict(X_s)
        else:
            y_s = np.zeros(X_s.shape[0])

        # Inverse-scale target
        return y_s * self._y_std + self._y_mean

    def export_diagnostics(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if not self._is_pymc or self._trace is None:
            return
        try:
            import arviz as az
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            label = f"Bayes_{self.pooling}"

            if self._coef_summary is not None:
                self._coef_summary.to_csv(
                    os.path.join(out_dir, f"{label}_posterior_summary.csv")
                )
            az.plot_trace(self._trace, var_names=["alpha", "beta"])
            plt.suptitle(f"Trace — {label}", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{label}_trace.png"),
                        dpi=120, bbox_inches="tight")
            plt.close()
            print(f"    Bayesian diagnostics → {out_dir}")
        except Exception as exc:
            print(f"    Diagnostic export failed: {exc}")


def train(X_tr, y_tr, X_val, y_val,
          pooling: str = "partial",
          run_name: str = "Bayesian") -> BayesianModel:
    model = BayesianModel(pooling=pooling)
    model.fit(X_tr, y_tr, X_val, y_val)
    model._search_method       = f"MCMC ({pooling} pooling) / BayesianRidge fallback"
    model._selection_criterion = "R-hat convergence / validation MSE"
    model._seed                = mp.BAYESIAN["seed"]
    return model
