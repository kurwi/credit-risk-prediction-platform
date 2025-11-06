"""
Hyperparameter optimization using Optuna.
"""

import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Automated hyperparameter tuning for credit risk models.
    """
    
    def __init__(self, n_trials: int = 50, cv_folds: int = 5):
        """
        Initialize optimizer.
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Cross-validation folds
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
        self.study = None
        
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting optimization with {self.n_trials} trials")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                    scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        self.study = optuna.create_study(direction='maximize', 
                                         study_name='credit_risk_optimization')
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        self.best_params['objective'] = 'binary:logistic'
        self.best_params['eval_metric'] = 'auc'
        self.best_params['random_state'] = 42
        
        logger.info(f"Optimization complete. Best AUC: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization trial history.
        
        Returns:
            DataFrame with trial results
        """
        if self.study is None:
            return pd.DataFrame()
        
        df = self.study.trials_dataframe()
        return df
