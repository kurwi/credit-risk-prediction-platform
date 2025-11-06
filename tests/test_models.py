"""
Test suite for credit risk prediction platform.
"""

import pytest
import pandas as pd
import numpy as np
from src.models import CreditRiskModel
from src.data import DataProcessor, DataGenerator
from src.validation import DataValidator


class TestCreditRiskModel:
    """Tests for CreditRiskModel."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = CreditRiskModel()
        assert model is not None
        assert model.optimal_threshold == 0.5
    
    def test_model_training(self):
        """Test model training."""
        df = DataGenerator.generate(n_samples=100)
        processor = DataProcessor()
        X, y = processor.fit_transform(df)
        
        model = CreditRiskModel()
        metrics = model.train(X, y)
        
        assert 'train_auc' in metrics
        assert metrics['train_auc'] > 0.5
        assert metrics['n_samples'] == 100
    
    def test_model_prediction(self):
        """Test model prediction."""
        df = DataGenerator.generate(n_samples=100)
        processor = DataProcessor()
        X, y = processor.fit_transform(df)
        
        model = CreditRiskModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        df = DataGenerator.generate(n_samples=200)
        processor = DataProcessor()
        X, y = processor.fit_transform(df)
        
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        model = CreditRiskModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert metrics['auc'] >= 0
        assert metrics['auc'] <= 1
    
    def test_threshold_optimization(self):
        """Test threshold optimization."""
        df = DataGenerator.generate(n_samples=100)
        processor = DataProcessor()
        X, y = processor.fit_transform(df)
        
        model = CreditRiskModel()
        model.train(X, y)
        
        threshold = model.optimize_threshold(X, y, cost_fn=1.0, cost_fp=0.1)
        assert 0 < threshold < 1
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        df = DataGenerator.generate(n_samples=100)
        processor = DataProcessor()
        X, y = processor.fit_transform(df)
        
        model = CreditRiskModel()
        model.train(X, y)
        
        importance_df = model.get_feature_importance()
        assert len(importance_df) == X.shape[1]
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


class TestDataProcessor:
    """Tests for DataProcessor."""
    
    def test_processor_fit_transform(self):
        """Test fit and transform."""
        df = DataGenerator.generate(n_samples=100)
        processor = DataProcessor()
        
        X, y = processor.fit_transform(df)
        
        assert processor.is_fitted
        assert len(X) == len(df)
        assert len(y) == len(df)
        assert X.shape[1] > 0
    
    def test_processor_transform_new_data(self):
        """Test transform on new data."""
        df_train = DataGenerator.generate(n_samples=100, random_state=42)
        df_test = DataGenerator.generate(n_samples=50, random_state=99)
        
        processor = DataProcessor()
        processor.fit(df_train)
        
        X_test, y_test = processor.transform(df_test)
        assert len(X_test) == 50
        assert len(y_test) == 50
    
    def test_feature_stats(self):
        """Test feature statistics."""
        df = DataGenerator.generate(n_samples=100)
        processor = DataProcessor()
        processor.fit(df)
        
        stats = processor.get_feature_stats(df)
        
        assert 'n_samples' in stats
        assert 'n_features' in stats
        assert stats['n_samples'] == 100


class TestDataGenerator:
    """Tests for DataGenerator."""
    
    def test_generate_data(self):
        """Test data generation."""
        df = DataGenerator.generate(n_samples=100)
        
        assert len(df) == 100
        assert 'target' in df.columns
        assert df['target'].isin([0, 1]).all()
    
    def test_generate_reproducible(self):
        """Test reproducibility with random seed."""
        df1 = DataGenerator.generate(n_samples=50, random_state=42)
        df2 = DataGenerator.generate(n_samples=50, random_state=42)
        
        pd.testing.assert_frame_equal(df1, df2)


class TestDataValidator:
    """Tests for DataValidator."""
    
    def test_validation_success(self):
        """Test validation on good data."""
        df = DataGenerator.generate(n_samples=100)
        validator = DataValidator()
        
        results = validator.validate(df)
        
        assert results['is_valid']
        assert len(results['errors']) == 0
    
    def test_missing_columns(self):
        """Test validation catches missing columns."""
        df = pd.DataFrame({'age': [25, 30], 'income': [50000, 60000]})
        validator = DataValidator()
        
        results = validator.validate(df)
        
        assert not results['is_valid']
        assert len(results['errors']) > 0
    
    def test_quality_report(self):
        """Test quality report generation."""
        df = DataGenerator.generate(n_samples=100)
        validator = DataValidator()
        
        report = validator.generate_quality_report(df)
        
        assert 'n_samples' in report
        assert 'n_features' in report
        assert report['n_samples'] == 100
