"""
Script to generate synthetic credit risk data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataGenerator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Generate synthetic credit risk dataset."""
    logger.info("Starting data generation")
    
    output_path = os.path.join('data', 'credit_data.csv')
    os.makedirs('data', exist_ok=True)
    
    df = DataGenerator.generate_and_save(
        filepath=output_path,
        n_samples=1000,
        random_state=42
    )
    
    logger.info(f"Generated {len(df)} records")
    logger.info(f"Features: {list(df.columns)}")
    logger.info(f"Default rate: {df['target'].mean():.2%}")
    logger.info(f"Data saved to {output_path}")
    
    print("\n" + "="*60)
    print("Data Generation Complete")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Default rate: {df['target'].mean():.2%}")
    print(f"Output: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
