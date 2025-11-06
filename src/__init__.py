"""
Credit Risk Prediction Platform

A production-ready credit risk assessment system with ML models,
data validation, API endpoints, and comprehensive analytics.
"""

__version__ = "1.0.0"
__author__ = "Portfolio Project"

from src.models import CreditRiskModel
from src.data import DataProcessor
from src.validation import DataValidator

__all__ = ["CreditRiskModel", "DataProcessor", "DataValidator"]
