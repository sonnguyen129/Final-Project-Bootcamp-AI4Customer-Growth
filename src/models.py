"""
Pydantic models for API request/response schemas

This module contains all data models used for API input validation and output serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class CLVMethod(str, Enum):
    BGNBD = "bgnbd"
    SURVIVAL = "survival"


class RetentionStrategy(str, Enum):
    HIGH_CLV_HIGH_CHURN = "high_clv_high_churn"
    HIGH_CHURN = "high_churn"
    LOW_P_ALIVE = "low_p_alive"


class ChurnRiskLabel(str, Enum):
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"


# ============================================================================
# Request Models
# ============================================================================

class ScoreCustomerRequest(BaseModel):
    """Request model for unified customer scoring endpoint"""
    customer_id: str = Field(..., description="Customer ID to score", example="C123")


class PredictChurnRequest(BaseModel):
    """Request model for churn prediction endpoint"""
    customer_id: str = Field(..., description="Customer ID", example="C123")
    horizon_days: int = Field(default=60, ge=1, le=365, description="Prediction horizon in days")


class PredictSurvivalRequest(BaseModel):
    """Request model for survival prediction endpoint"""
    customer_id: str = Field(..., description="Customer ID", example="C123")


class EstimateCLVRequest(BaseModel):
    """Request model for CLV estimation endpoint"""
    customer_id: str = Field(..., description="Customer ID", example="C123")
    method: CLVMethod = Field(default=CLVMethod.BGNBD, description="CLV calculation method")
    horizon_months: int = Field(default=12, ge=1, le=36, description="CLV horizon in months")


class RankCustomersRequest(BaseModel):
    """Request model for customer retention ranking endpoint"""
    top_k: int = Field(default=100, ge=1, le=10000, description="Number of top customers to return")
    strategy: RetentionStrategy = Field(
        default=RetentionStrategy.HIGH_CLV_HIGH_CHURN, 
        description="Retention prioritization strategy"
    )


# ============================================================================
# Response Models
# ============================================================================

class SurvivalPoint(BaseModel):
    """Single point on survival curve"""
    day: int
    prob: float


class ScoreCustomerResponse(BaseModel):
    """Response model for unified customer scoring"""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1, description="Classification-based churn probability")
    p_alive: float = Field(..., ge=0, le=1, description="BG-NBD probability of being alive")
    expected_remaining_lifetime: float = Field(..., ge=0, description="Expected remaining lifetime in months")
    clv_bgnbd: float = Field(..., ge=0, description="CLV from BG-NBD + Gamma-Gamma method")
    clv_survival: float = Field(..., ge=0, description="CLV from Survival Analysis method")


class PredictChurnResponse(BaseModel):
    """Response model for churn prediction"""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_label: ChurnRiskLabel


class PredictSurvivalResponse(BaseModel):
    """Response model for survival prediction"""
    customer_id: str
    survival_curve: List[SurvivalPoint]
    expected_remaining_lifetime: float = Field(..., ge=0, description="Expected remaining lifetime in months")


class EstimateCLVResponse(BaseModel):
    """Response model for CLV estimation"""
    customer_id: str
    method: CLVMethod
    clv: float = Field(..., ge=0)
    horizon_months: int


class RankedCustomer(BaseModel):
    """Single customer in retention ranking"""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    clv: float = Field(..., ge=0)
    priority_score: float = Field(..., ge=0)


class RankCustomersResponse(BaseModel):
    """Response model for customer retention ranking"""
    strategy: RetentionStrategy
    total_customers: int
    customers: List[RankedCustomer]


# ============================================================================
# Error Response Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None


class CustomerNotFoundError(ErrorResponse):
    """Customer not found error"""
    error: str = "customer_not_found"
    customer_id: str
