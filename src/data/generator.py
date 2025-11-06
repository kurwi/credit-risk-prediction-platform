"""
Synthetic credit risk data generation for testing and demonstrations.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Generate realistic synthetic credit risk data.
    """
    
    @staticmethod
    def generate(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic credit application data.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with credit features and target
        """
        np.random.seed(random_state)
        logger.info(f"Generating {n_samples} synthetic credit records")
        
        age = np.random.randint(18, 75, n_samples)
        income = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)
        credit_history_months = np.random.randint(0, 360, n_samples)
        existing_loans = np.random.poisson(lam=1.5, size=n_samples)
        debt_to_income_ratio = np.random.uniform(0, 0.6, n_samples)
        
        employment_status = np.random.choice(
            ['Employed', 'Self-Employed', 'Unemployed', 'Retired'],
            size=n_samples,
            p=[0.6, 0.2, 0.1, 0.1]
        )
        
        housing_status = np.random.choice(
            ['Own', 'Rent', 'Mortgage'],
            size=n_samples,
            p=[0.3, 0.4, 0.3]
        )
        
        loan_purpose = np.random.choice(
            ['Personal', 'Auto', 'Home', 'Education', 'Business'],
            size=n_samples,
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        
        loan_amount = np.random.lognormal(mean=9.5, sigma=0.7, size=n_samples)
        
        credit_score = np.random.normal(loc=650, scale=100, size=n_samples)
        credit_score = np.clip(credit_score, 300, 850).astype(int)
        
        # Build a realistic risk score that produces ~15-20% default rate
        risk_score = (
            -0.02 * age
            - 0.00002 * income
            - 0.005 * credit_history_months
            + 0.3 * existing_loans
            + 3.0 * debt_to_income_ratio
            - 0.006 * credit_score
            + 0.0001 * loan_amount
            + np.random.normal(0, 0.8, n_samples)
        )
        
        default_proba = 1 / (1 + np.exp(-risk_score))
        target = (default_proba > 0.5).astype(int)
        
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'credit_history_months': credit_history_months,
            'existing_loans': existing_loans,
            'debt_to_income_ratio': debt_to_income_ratio,
            'employment_status': employment_status,
            'housing_status': housing_status,
            'loan_purpose': loan_purpose,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'target': target
        })
        
        logger.info(f"Generated data: {target.sum()} defaults ({100*target.mean():.1f}%)")
        
        return df
    
    @staticmethod
    def generate_and_save(
        filepath: str,
        n_samples: int = 1000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate and save synthetic data to CSV.
        
        Args:
            filepath: Output CSV path
            n_samples: Number of samples
            random_state: Random seed
            
        Returns:
            Generated DataFrame
        """
        df = DataGenerator.generate(n_samples, random_state)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return df
