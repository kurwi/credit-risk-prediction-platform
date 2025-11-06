"""
Train model once and save it for production use in the app.
Run this script to generate a pre-trained model.
"""
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataGenerator, DataProcessor
from src.models import CreditRiskModel
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate data, train model, optimize threshold, and save everything."""
    logger.info("Generating synthetic credit data...")
    df = DataGenerator.generate(n_samples=5000, random_state=42)
    
    logger.info("Processing data...")
    processor = DataProcessor()
    X, y = processor.fit_transform(df, target_col='target')
    
    # Split for training and threshold optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Training XGBoost model...")
    model = CreditRiskModel()
    model.train(X_train, y_train)
    
    logger.info("Optimizing decision threshold...")
    # Cost: False Negative (approving bad loan) = 10x more costly than False Positive (rejecting good client)
    model.optimize_threshold(X_val, y_val, cost_fn=10.0, cost_fp=1.0)
    
    logger.info(f"Optimal threshold: {model.optimal_threshold:.3f}")
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    logger.info(f"Validation AUC: {metrics['auc']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    
    # Save everything
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/preprocessor', exist_ok=True)
    
    logger.info("Saving model and preprocessor...")
    model.save_with_metrics('models/credit_risk_model.pkl', 'models/model_metadata.json', metrics)
    processor.save('models/preprocessor/scaler.pkl', 'models/preprocessor/encoders.pkl')
    
    # Save feature names for reference
    import json
    with open('models/feature_info.json', 'w') as f:
        json.dump({
            'feature_names': processor.feature_names,
            'categorical_features': processor.categorical_cols,
            'numerical_features': processor.numerical_cols
        }, f, indent=2)
    
    logger.info("Model and preprocessor saved successfully!")
    logger.info("   - Model: models/credit_risk_model.pkl")
    logger.info("   - Metadata: models/model_metadata.json")
    logger.info("   - Preprocessor: models/preprocessor/")
    logger.info("   - Feature info: models/feature_info.json")

if __name__ == '__main__':
    main()
