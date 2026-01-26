"""
Customer Churn & CLV Prediction - Source Module

This package contains all components for production-ready churn prediction
and CLV estimation.

Modules:
- api_service: FastAPI REST endpoints
- scoring_engine: Core prediction business logic
- model_loader: Model loading and caching
- models: Pydantic data models for API
- feature_engineer: Feature engineering pipeline
- coxph_feature_prepation: Cox PH feature preparation
"""

from .model_loader import ModelLoader, get_model_loader, save_models
from .scoring_engine import ScoringEngine, get_scoring_engine
from .models import (
    CLVMethod,
    RetentionStrategy,
    ChurnRiskLabel,
    ScoreCustomerRequest,
    ScoreCustomerResponse,
)

__version__ = "1.0.0"
__all__ = [
    "ModelLoader",
    "get_model_loader",
    "save_models",
    "ScoringEngine",
    "get_scoring_engine",
    "CLVMethod",
    "RetentionStrategy",
    "ChurnRiskLabel",
]
