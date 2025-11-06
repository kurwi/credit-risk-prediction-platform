"""
Train credit risk prediction model.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CreditRiskModel, ModelOptimizer
from src.data import DataProcessor, DataGenerator
from src.validation import DataValidator
import pandas as pd
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Complete model training pipeline:
    1. Load/generate data
    2. Validate data quality
    3. Preprocess features
    4. Train model (optional: with hyperparameter optimization)
    5. Evaluate performance
    6. Save model and artifacts
    """
    
    logger.info("="*60)
    logger.info("Credit Risk Model Training Pipeline")
    logger.info("="*60)
    
    data_path = os.path.join('data', 'credit_data.csv')
    
    if not os.path.exists(data_path):
        logger.info("Generating synthetic data...")
        df = DataGenerator.generate_and_save(data_path, n_samples=1000)
    else:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    
    logger.info("Step 1: Data Validation")
    validator = DataValidator()
    validation_results = validator.validate(df)
    
    if not validation_results['is_valid']:
        logger.error("Data validation failed:")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")
        return
    
    logger.info("Data validation passed")
    
    logger.info("Step 2: Data Preprocessing")
    processor = DataProcessor()
    X, y = processor.fit_transform(df, target_col='target')
    
    logger.info(f"Processed {len(X)} samples with {X.shape[1]} features")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Step 3: Model Training")
    
    use_optimization = False
    
    if use_optimization:
        logger.info("Running hyperparameter optimization...")
        optimizer = ModelOptimizer(n_trials=20, cv_folds=3)
        best_params = optimizer.optimize(X_train, y_train)
        model = CreditRiskModel(params=best_params)
    else:
        logger.info("Using default parameters")
        model = CreditRiskModel()
    
    train_metrics = model.train(X_train, y_train)
    logger.info(f"Training AUC: {train_metrics['train_auc']:.4f}")
    
    logger.info("Step 4: Threshold Optimization")
    optimal_threshold = model.optimize_threshold(
        X_train, y_train,
        cost_fn=1.0,
        cost_fp=0.1
    )
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    
    logger.info("Step 5: Model Evaluation")
    test_metrics = model.evaluate(X_test, y_test)
    
    logger.info("Test Performance:")
    logger.info(f"  AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1: {test_metrics['f1']:.4f}")
    
    logger.info("Step 6: Saving Model")
    os.makedirs('models', exist_ok=True)
    
    model.save(
        model_path=os.path.join('models', 'credit_risk_model.pkl'),
        metadata_path=os.path.join('models', 'model_metadata.json')
    )
    
    processor.save(
        scaler_path=os.path.join('models', 'scaler.pkl'),
        encoders_path=os.path.join('models', 'label_encoders.pkl')
    )
    
    results_path = os.path.join('models', 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'optimal_threshold': float(optimal_threshold)
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    logger.info("Step 7: Feature Importance")
    importance_df = model.get_feature_importance()
    importance_path = os.path.join('models', 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    logger.info("Top 5 features:")
    for idx, row in importance_df.head().iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Model saved to models/")
    print("="*60)


if __name__ == "__main__":
    main()
