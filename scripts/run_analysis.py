"""
Run complete analysis and generate comprehensive report.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CreditRiskModel
from src.data import DataProcessor
from src.validation import DataValidator
import pandas as pd
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run end-to-end analysis pipeline."""
    
    logger.info("="*60)
    logger.info("Credit Risk Analysis Pipeline")
    logger.info("="*60)
    
    data_path = os.path.join('data', 'credit_data.csv')
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Run scripts/generate_data.py first")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")
    
    validator = DataValidator()
    quality_report = validator.generate_quality_report(df)
    
    logger.info("Data Quality Report:")
    logger.info(f"  Samples: {quality_report['n_samples']}")
    logger.info(f"  Features: {quality_report['n_features']}")
    logger.info(f"  Duplicates: {quality_report['duplicate_rows']}")
    logger.info(f"  Default rate: {quality_report.get('class_balance', {}).get(1, 0):.2%}")
    
    processor = DataProcessor()
    X, y = processor.fit_transform(df)
    
    model_path = os.path.join('models', 'credit_risk_model.pkl')
    metadata_path = os.path.join('models', 'model_metadata.json')
    
    if not os.path.exists(model_path):
        logger.error("Trained model not found")
        logger.info("Run scripts/train_model.py first")
        return
    
    logger.info("Loading trained model...")
    model = CreditRiskModel.load(model_path, metadata_path)
    
    logger.info("Generating predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    df['predicted_default'] = predictions
    df['default_probability'] = probabilities
    df['risk_score'] = (probabilities * 100).astype(int)
    
    high_risk = df[df['default_probability'] > 0.7]
    logger.info(f"High risk applicants: {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)")
    
    os.makedirs('data/processed', exist_ok=True)
    output_path = os.path.join('data', 'processed', 'scored_applications.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Scored data saved to {output_path}")
    
    report = {
        'total_applications': len(df),
        'high_risk_count': int(len(high_risk)),
        'high_risk_percentage': float(len(high_risk) / len(df) * 100),
        'average_risk_score': float(df['risk_score'].mean()),
        'median_risk_score': float(df['risk_score'].median()),
        'quality_report': quality_report
    }
    
    report_path = os.path.join('data', 'processed', 'analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Analysis report saved to {report_path}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Total Applications: {len(df)}")
    print(f"High Risk: {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)")
    print(f"Average Risk Score: {df['risk_score'].mean():.1f}/100")
    print(f"Output: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
