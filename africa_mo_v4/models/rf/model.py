"""models/rf/model.py — Random Forest with Optuna hyperparameter search."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tuning.optuna_search import tune_random_forest
import config.model_params as mp


def train(X_tr, y_tr, X_val, y_val,
          run_name: str = "RF") -> RandomForestRegressor:
    best = tune_random_forest(X_tr, y_tr, X_val, y_val, run_name=run_name)
    model = RandomForestRegressor(
        **best, random_state=mp.RF["seed"], n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    model._best_params    = best
    model._search_method  = "Optuna TPE"
    model._n_trials       = mp.RF["n_trials"]
    model._selection_criterion = "RMSE on inner validation slice"
    model._seed           = mp.RF["seed"]
    return model
