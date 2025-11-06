"""
Credit risk prediction model with XGBoost and interpretability.
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Production-ready credit risk prediction model.
    
    Features:
    - XGBoost classifier with optimized hyperparameters
    - Cost-sensitive threshold optimization
    - SHAP-based model interpretability
    - Comprehensive performance metrics
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the credit risk model.
        Args:
            params (Optional[Dict]): XGBoost hyperparameters. If None, uses defaults.
        """
        """
        Initialize the credit risk model.
        
        Args:
            params: XGBoost hyperparameters. If None, uses defaults.
        """
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            # Explicitly set base_score for XGBoost>=3 which requires (0,1)
            'base_score': 0.5,
            # Sensible defaults for speed/stability
            'tree_method': 'hist',
            'n_jobs': -1
        }
        self.params = params if params is not None else default_params
        # Ensure base_score is valid for logistic objective
        try:
            bs = float(self.params.get('base_score', 0.5))
            if not (0.0 < bs < 1.0):
                logger.warning(
                    "Invalid base_score %s for logistic loss; falling back to 0.5",
                    self.params.get('base_score')
                )
                self.params['base_score'] = 0.5
        except Exception:
            self.params['base_score'] = 0.5
        self.model = xgb.XGBClassifier(**self.params)
        self.optimal_threshold = 0.5
        self.feature_names = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the credit risk model.
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training labels (0=good, 1=default)
        Returns:
            Dict: Training metrics
        """
        """
        Train the credit risk model.
        
        Args:
            X: Training features
            y: Training labels (0=good, 1=default)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training credit risk model on {len(X)} samples")
        self.feature_names = list(X.columns)
        
        self.model.fit(X, y)
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'train_auc': roc_auc_score(y, y_pred_proba),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        logger.info(f"Training complete. AUC: {metrics['train_auc']:.4f}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict credit risk class (0=good, 1=default).
        Args:
            X (pd.DataFrame): Feature matrix
        Returns:
            np.ndarray: Binary predictions
        """
        """
        Predict credit risk class (0=good, 1=default).
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= self.optimal_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default.
        Args:
            X (pd.DataFrame): Feature matrix
        Returns:
            np.ndarray: Default probabilities
        """
        """
        Predict probability of default.
        
        Args:
            X: Feature matrix
            
        Returns:
            Default probabilities
        """
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Comprehensive model evaluation.
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True labels
        Returns:
            Dict: Performance metrics
        """
        """
        Comprehensive model evaluation.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary with performance metrics
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # AUC can fail when y has a single class; guard it
        try:
            auc_val = roc_auc_score(y, y_pred_proba)
        except Exception:
            auc_val = float('nan')

        metrics = {
            'auc': auc_val,
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
            'threshold': self.optimal_threshold
        }
        
        logger.info(f"Evaluation - AUC: {metrics['auc']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def optimize_threshold(self, X: pd.DataFrame, y: pd.Series, 
                          cost_fn: float = 1.0, cost_fp: float = 0.1) -> float:
        """
        Optimize classification threshold based on cost matrix.
        Args:
            X (pd.DataFrame): Validation features
            y (pd.Series): True labels
            cost_fn (float): Cost of false negative (missed default)
            cost_fp (float): Cost of false positive (rejected good customer)
        Returns:
            float: Optimal threshold
        """
        """
        Optimize classification threshold based on cost matrix.
        
        Args:
            X: Validation features
            y: True labels
            cost_fn: Cost of false negative (missed default)
            cost_fp: Cost of false positive (rejected good customer)
            
        Returns:
            Optimal threshold
        """
        y_pred_proba = self.predict_proba(X)
        
        thresholds = np.linspace(0.1, 0.9, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            # Ensure a 2x2 confusion matrix even if one class is missing
            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            total_cost = (fn * cost_fn + fp * cost_fp) / len(y)
            costs.append(total_cost)
        
        optimal_idx = np.argmin(costs)
        self.optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold: {self.optimal_threshold:.4f} "
                   f"(min cost: {costs[optimal_idx]:.4f})")
        
        return self.optimal_threshold
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        Returns:
            pd.DataFrame: Features and importance scores
        """
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and importance scores
        """
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Save model and metadata to disk.
        Args:
            model_path (str): Path to save model pickle
            metadata_path (Optional[str]): Path to save metadata JSON
        Raises:
            IOError: If saving fails
        """
        try:
            joblib.dump(self.model, model_path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError(f"Failed to save model: {e}")
        if metadata_path:
            metadata = {
                'threshold': float(self.optimal_threshold),
                'feature_names': self.feature_names,
                'params': self.params
            }
            import json
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
                raise IOError(f"Failed to save metadata: {e}")
        logger.info(f"Model saved to {model_path}")
        """
        Save model and metadata to disk.
        
        Args:
            model_path: Path to save model pickle
            metadata_path: Path to save metadata JSON
        """
        joblib.dump(self.model, model_path)
        
        if metadata_path:
            metadata = {
                'threshold': float(self.optimal_threshold),
                'feature_names': self.feature_names,
                'params': self.params
            }
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def save_with_metrics(self, model_path: str, metadata_path: str, metrics: Dict):
        """
        Save model, metadata, and evaluation metrics to disk.
        
        Args:
            model_path: Path to save model pickle
            metadata_path: Path to save metadata JSON
            metrics: Evaluation metrics dictionary (auc, precision, recall, f1)
        """
        joblib.dump(self.model, model_path)
        
        metadata = {
            'threshold': float(self.optimal_threshold),
            'feature_names': self.feature_names,
            'params': self.params,
            'auc': float(metrics.get('auc', 0)),
            'precision': float(metrics.get('precision', 0)),
            'recall': float(metrics.get('recall', 0)),
            'f1': float(metrics.get('f1', 0))
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model and metrics saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None):
        """
        Load model from disk.
        Args:
            model_path (str): Path to model pickle
            metadata_path (Optional[str]): Path to metadata JSON
        Returns:
            CreditRiskModel: Loaded instance
        Raises:
            IOError: If loading fails
        """
        instance = cls()
        try:
            instance.model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise IOError(f"Failed to load model: {e}")
        if metadata_path:
            import json
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                instance.optimal_threshold = metadata.get('threshold', 0.5)
                instance.feature_names = metadata.get('feature_names')
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                raise IOError(f"Failed to load metadata: {e}")
        logger.info(f"Model loaded from {model_path}")
        return instance

    def explain(self, X: pd.DataFrame, nsamples: int = 100) -> Dict:
        """
        Generate SHAP-based explainability for predictions.
        Args:
            X (pd.DataFrame): Input features
            nsamples (int): Number of samples for summary plot
        Returns:
            Dict: SHAP values and summary plot (as matplotlib figure)
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP is not installed. Run 'pip install shap'.")
            raise ImportError("SHAP is not installed. Run 'pip install shap'.")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        summary_plot = None
        try:
            import matplotlib.pyplot as plt
            summary_plot = shap.summary_plot(shap_values, X, show=False, max_display=nsamples)
            fig = plt.gcf()
        except Exception as e:
            logger.warning(f"Could not generate SHAP summary plot: {e}")
            fig = None
        return {'shap_values': shap_values, 'summary_plot': fig}
        """
        Load model from disk.
        
        Args:
            model_path: Path to model pickle
            metadata_path: Path to metadata JSON
            
        Returns:
            Loaded CreditRiskModel instance
        """
        instance = cls()
        instance.model = joblib.load(model_path)
        
        if metadata_path:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            instance.optimal_threshold = metadata.get('threshold', 0.5)
            instance.feature_names = metadata.get('feature_names')
        
        logger.info(f"Model loaded from {model_path}")
        return instance
