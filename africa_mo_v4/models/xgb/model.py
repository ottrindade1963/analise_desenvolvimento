"""models/xgb/model.py — XGBoost with Optuna hyperparameter search."""
import numpy as np
from tuning.optuna_search import tune_xgboost
import config.model_params as mp


def train(X_tr, y_tr, X_val, y_val,
          run_name: str = "XGB"):
    best = tune_xgboost(X_tr, y_tr, X_val, y_val, run_name=run_name)

    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            **best,
            n_estimators=mp.XGB["n_estimators"],
            early_stopping_rounds=mp.XGB["early_stopping_rounds"],
            random_state=mp.XGB["seed"],
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=300, random_state=42)
        model.fit(X_tr, y_tr)

    model._best_params         = best
    model._search_method       = "Optuna TPE"
    model._n_trials            = mp.XGB["n_trials"]
    model._selection_criterion = "RMSE on inner validation slice"
    model._seed                = mp.XGB["seed"]
    return model
