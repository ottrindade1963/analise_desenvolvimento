# Pipeline Executive Summary
*Generated: 2026-07-21 15:32*

**Project**: Análise Industrial — África e Médio Oriente v4.0

## Method
Walk-forward cross-validation (5 folds) with fold-level MICE imputation, StandardScaler, and PCA applied exclusively on training data. Optuna TPE hyperparameter search (50 trials per model). Ablation study over 5 governance specifications.

## Key results
- **best_RMSE_single_fold (mínimo entre as linhas de fold individuais)**: 1.6940
- **best_model_single_fold**: LSTM
- **best_spec_single_fold**: A1_WDI_only
- **best_RMSE_mean_per_group (média por especificação×modelo — comparável à tabela de desempenho)**: 2.2337
- **best_model_mean_per_group**: LSTM
- **best_spec_mean_per_group**: A1_WDI_only
- **n_models**: 6
- **n_specs**: 5
- **n_folds**: 5