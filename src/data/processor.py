"""
Data preprocessing and feature engineering for credit risk models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List, Optional
import logging
import joblib

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data preprocessing pipeline for credit risk modeling.
    """
    
    def __init__(self):
        """Initialize processor with empty state."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, target_col: str = 'target') -> 'DataProcessor':
        """
        Fit preprocessing transformers on training data.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column
            
        Returns:
            Self (fitted processor)
        """
        logger.info(f"Fitting data processor on {len(df)} samples")
        
        self.categorical_cols = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        if target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)
        
        self.numerical_cols = df.select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']
        ).columns.tolist()
        
        if target_col in self.numerical_cols:
            self.numerical_cols.remove(target_col)
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
        
        if self.numerical_cols:
            self.scaler.fit(df[self.numerical_cols])
        
        self.feature_names = self.categorical_cols + self.numerical_cols
        self.is_fitted = True
        
        logger.info(f"Processor fitted: {len(self.categorical_cols)} categorical, "
                   f"{len(self.numerical_cols)} numerical features")
        
        return self
    
    def transform(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform data using fitted preprocessors.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (transformed features, target if present)
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        df = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        if self.numerical_cols:
            df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
        
        if target_col in df.columns:
            y = df[target_col]
            X = df[self.feature_names]
            return X, y
        else:
            return df[self.feature_names], None
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (transformed features, target)
        """
        self.fit(df, target_col)
        return self.transform(df, target_col)
    
    def inverse_transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert encoded categoricals back to original values.
        
        Args:
            df: DataFrame with encoded categoricals
            
        Returns:
            DataFrame with decoded categoricals
        """
        df = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = self.label_encoders[col].inverse_transform(
                    df[col].astype(int)
                )
        
        return df
    
    def save(self, scaler_path: str, encoders_path: str):
        """
        Save fitted preprocessors.
        
        Args:
            scaler_path: Path to save scaler
            encoders_path: Path to save label encoders
        """
        # Save all necessary state
        state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols
        }
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(state, encoders_path)
        logger.info("Preprocessors saved")
    
    @classmethod
    def load(cls, scaler_path: str, encoders_path: str) -> 'DataProcessor':
        """
        Load fitted preprocessors.
        
        Args:
            scaler_path: Path to scaler pickle
            encoders_path: Path to encoders pickle
            
        Returns:
            Loaded DataProcessor
        """
        instance = cls()
        instance.scaler = joblib.load(scaler_path)
        
        # Load state (backward compatible)
        loaded = joblib.load(encoders_path)
        if isinstance(loaded, dict) and 'label_encoders' in loaded:
            # New format with full state
            instance.label_encoders = loaded['label_encoders']
            instance.feature_names = loaded.get('feature_names')
            instance.categorical_cols = loaded.get('categorical_cols', list(instance.label_encoders.keys()))
            instance.numerical_cols = loaded.get('numerical_cols', [])
        else:
            # Old format (just encoders dict)
            instance.label_encoders = loaded
            instance.categorical_cols = list(instance.label_encoders.keys())
            instance.feature_names = None
            instance.numerical_cols = []
        
        instance.is_fitted = True
        logger.info("Preprocessors loaded")
        return instance
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive feature statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'n_samples': len(df),
            'n_features': len(self.feature_names),
            'categorical_features': len(self.categorical_cols),
            'numerical_features': len(self.numerical_cols),
            'missing_values': df[self.feature_names].isnull().sum().to_dict(),
            'categorical_cardinality': {
                col: len(self.label_encoders[col].classes_)
                for col in self.categorical_cols
            }
        }
        
        return stats
