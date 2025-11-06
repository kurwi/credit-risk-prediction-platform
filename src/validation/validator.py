"""
Data quality validation for credit risk datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for credit risk modeling.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.required_columns = [
            'age', 'income', 'credit_score', 'target'
        ]
        self.validation_results = {}
        
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Run all validation checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Running data validation checks")
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        results.update(self._check_required_columns(df))
        results.update(self._check_missing_values(df))
        results.update(self._check_data_types(df))
        results.update(self._check_value_ranges(df))
        results.update(self._check_target_distribution(df))
        
        results['is_valid'] = len(results['errors']) == 0
        
        if results['is_valid']:
            logger.info("Validation passed")
        else:
            logger.warning(f"Validation failed with {len(results['errors'])} errors")
        
        self.validation_results = results
        return results
    
    def _check_required_columns(self, df: pd.DataFrame) -> Dict:
        """Check if required columns are present."""
        missing = [col for col in self.required_columns if col not in df.columns]
        
        if missing:
            return {
                'errors': [f"Missing required columns: {missing}"]
            }
        return {}
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values."""
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            return {
                'warnings': [
                    f"Missing values in {len(missing_cols)} columns: "
                    f"{missing_cols.to_dict()}"
                ]
            }
        return {}
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """Validate data types."""
        errors = []
        
        numeric_cols = ['age', 'income', 'credit_score']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric")
        
        if errors:
            return {'errors': errors}
        return {}
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check if values are within expected ranges."""
        warnings = []
        
        if 'age' in df.columns:
            if (df['age'] < 18).any() or (df['age'] > 120).any():
                warnings.append("Age values outside expected range [18, 120]")
        
        if 'credit_score' in df.columns:
            if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
                warnings.append("Credit score outside expected range [300, 850]")
        
        if 'income' in df.columns:
            if (df['income'] < 0).any():
                warnings.append("Negative income values detected")
        
        if warnings:
            return {'warnings': warnings}
        return {}
    
    def _check_target_distribution(self, df: pd.DataFrame) -> Dict:
        """Check target variable distribution."""
        if 'target' not in df.columns:
            return {}
        
        target_counts = df['target'].value_counts()
        default_rate = df['target'].mean()
        
        stats = {
            'target_distribution': target_counts.to_dict(),
            'default_rate': float(default_rate)
        }
        
        warnings = []
        if default_rate < 0.05 or default_rate > 0.5:
            warnings.append(
                f"Unusual default rate: {default_rate:.1%}. "
                "Check data quality or consider class balancing."
            )
        
        return {'stats': stats, 'warnings': warnings}
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        self.validate(df)
        
        report = {
            'n_samples': len(df),
            'n_features': df.shape[1],
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'validation_results': self.validation_results
        }
        
        if 'target' in df.columns:
            report['class_balance'] = df['target'].value_counts(normalize=True).to_dict()
        
        return report
