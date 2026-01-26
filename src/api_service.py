"""
Customer Churn & CLV API Service

This module provides REST API endpoints for customer scoring, churn prediction,
survival analysis, and CLV estimation.

Endpoints:
- POST /score_customer - Unified customer scoring
- POST /predict_churn - Churn prediction as classification  
- POST /predict_survival - Time-to-churn prediction
- POST /estimate_clv - CLV estimation
- POST /rank_customers_for_retention - Retention prioritization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional

from .models import (
    # Request models
    ScoreCustomerRequest,
    PredictChurnRequest,
    PredictSurvivalRequest,
    EstimateCLVRequest,
    RankCustomersRequest,
    # Response models
    ScoreCustomerResponse,
    PredictChurnResponse,
    PredictSurvivalResponse,
    EstimateCLVResponse,
    RankCustomersResponse,
    CustomerNotFoundError,
    # Enums
    CLVMethod,
    RetentionStrategy,
)
from .scoring_engine import ScoringEngine, get_scoring_engine, reset_scoring_engine
from .model_loader import get_model_loader, reset_model_loader
from .data_processor import get_data_processor, reset_data_processor


# ============================================================================
# Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Load models and data
    print("🚀 Starting Customer Scoring API...")
    
    try:
        # Load models
        print("\n1️⃣ Loading trained models...")
        loader = get_model_loader()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("⚠️ API will start but predictions may fail until models are available.")
    
    try:
        # Load CSV data
        print("\n2️⃣ Loading customer data from CSV files...")
        processor = get_data_processor()
        print("✅ Data loaded successfully!")
        print(f"   👥 Customers: {len(processor.get_all_customer_ids()):,}")
        print(f"   📅 Observation date: {processor.observation_date.date()}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        print("⚠️ API will start but predictions may fail until data is available.")
    
    print("\n✅ API ready to serve requests!\n")
    
    yield
    
    # Shutdown: Cleanup
    print("🛑 Shutting down API...")
    reset_model_loader()
    reset_data_processor()
    reset_scoring_engine()


app = FastAPI(
    title="Customer Churn & CLV API",
    description="""
    Production-ready API for customer scoring, churn prediction, and CLV estimation.
    
    ## Features:
    - **Unified Customer Scoring**: Get all metrics for a customer in one call
    - **Churn Prediction**: Classification-based churn risk prediction
    - **Survival Analysis**: Time-to-churn prediction with survival curves
    - **CLV Estimation**: Customer lifetime value using BG-NBD or Survival methods
    - **Retention Ranking**: Prioritize customers for retention campaigns
    
    ## Models Used:
    - Gradient Boosting Classifier (churn classification)
    - BG-NBD Model (probability alive, expected transactions)
    - Gamma-Gamma Model (expected monetary value)
    - Cox Proportional Hazards Model (survival analysis)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Root & Health Check
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - Welcome message and API info."""
    return {
        "message": "🚀 Customer Churn & CLV API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /score_customer",
            "POST /predict_churn",
            "POST /predict_survival",
            "POST /estimate_clv",
            "POST /rank_customers_for_retention"
        ]
    }


@app.get("/health")
async def health_check():
    """Check API health and model status."""
    try:
        loader = get_model_loader()
        processor = get_data_processor()
        
        models_loaded = {
            "classification": loader.classification_model is not None,
            "bgf": loader.bgf_model is not None,
            "ggf": loader.ggf_model is not None,
            "cph": loader.cph_model is not None,
        }
        
        data_loaded = {
            "customers": processor.customers is not None,
            "transactions": processor.transactions is not None,
            "customer_count": len(processor.get_all_customer_ids()) if processor.customers is not None else 0
        }
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "data_loaded": data_loaded,
            "all_ready": all(models_loaded.values()) and all(v for k, v in data_loaded.items() if k != "customer_count")
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/score_customer", response_model=ScoreCustomerResponse)
async def score_customer(request: ScoreCustomerRequest):
    """
    Unified customer scoring endpoint.
    
    Combines outputs from:
    - Classification churn model
    - BG-NBD probabilistic model  
    - Survival analysis
    - CLV modeling (both methods)
    
    Returns comprehensive customer risk and value metrics.
    """
    # Create fresh instances for each request to avoid state issues
    processor = get_data_processor()
    
    # Check if customer exists
    if not processor.customer_exists(request.customer_id):
        raise HTTPException(
            status_code=404,
            detail=CustomerNotFoundError(
                customer_id=request.customer_id,
                detail=f"Customer {request.customer_id} not found in database"
            ).model_dump()
        )
    
    # Create engine (uses global singletons for loader and processor)
    engine = ScoringEngine()
    
    # Get customer score
    score = engine.score_customer(request.customer_id)
    
    if score is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate customer score"
        )
    
    return ScoreCustomerResponse(
        customer_id=score.customer_id,
        churn_probability=round(score.churn_probability, 4),
        p_alive=round(score.p_alive, 4),
        expected_remaining_lifetime=round(score.expected_remaining_lifetime, 2),
        clv_bgnbd=round(score.clv_bgnbd, 2),
        clv_survival=round(score.clv_survival, 2)
    )


@app.post("/predict_churn", response_model=PredictChurnResponse)
async def predict_churn(request: PredictChurnRequest):
    """
    Churn prediction as classification.
    
    Predicts the probability of customer churn within the specified horizon
    and assigns a risk label (low_risk, medium_risk, high_risk, critical_risk).
    """
    processor = get_data_processor()
    
    # Check if customer exists
    if not processor.customer_exists(request.customer_id):
        raise HTTPException(
            status_code=404,
            detail=CustomerNotFoundError(
                customer_id=request.customer_id,
                detail=f"Customer {request.customer_id} not found in database"
            ).model_dump()
        )
    
    # Create fresh engine for this request
    engine = ScoringEngine()
    
    result = engine.predict_churn(request.customer_id, request.horizon_days)
    
    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to predict churn"
        )
    
    churn_prob, churn_label = result
    
    return PredictChurnResponse(
        customer_id=request.customer_id,
        churn_probability=round(churn_prob, 4),
        churn_label=churn_label
    )


@app.post("/predict_survival", response_model=PredictSurvivalResponse)
async def predict_survival(request: PredictSurvivalRequest):
    """
    Time-to-churn prediction using survival analysis.
    
    Returns a survival curve showing probability of customer remaining active
    at different time points, along with expected remaining lifetime.
    """
    processor = get_data_processor()
    
    # Check if customer exists
    if not processor.customer_exists(request.customer_id):
        raise HTTPException(
            status_code=404,
            detail=CustomerNotFoundError(
                customer_id=request.customer_id,
                detail=f"Customer {request.customer_id} not found in database"
            ).model_dump()
        )
    
    # Create fresh engine for this request
    engine = ScoringEngine()
    
    result = engine.predict_survival_curve(request.customer_id)
    
    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to predict survival curve"
        )
    
    survival_curve, expected_lifetime = result
    
    return PredictSurvivalResponse(
        customer_id=request.customer_id,
        survival_curve=survival_curve,
        expected_remaining_lifetime=round(expected_lifetime, 2)
    )


@app.post("/estimate_clv", response_model=EstimateCLVResponse)
async def estimate_clv(request: EstimateCLVRequest):
    """
    Customer Lifetime Value estimation.
    
    Supports two methods:
    - **bgnbd**: BG-NBD + Gamma-Gamma probabilistic approach
    - **survival**: Survival analysis with time-dependent CLV
    """
    processor = get_data_processor()
    
    # Check if customer exists
    if not processor.customer_exists(request.customer_id):
        raise HTTPException(
            status_code=404,
            detail=CustomerNotFoundError(
                customer_id=request.customer_id,
                detail=f"Customer {request.customer_id} not found in database"
            ).model_dump()
        )
    
    # Create fresh engine for this request
    engine = ScoringEngine()
    
    clv = engine.estimate_clv(
        request.customer_id, 
        request.method,
        request.horizon_months
    )
    
    if clv is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to estimate CLV"
        )
    
    return EstimateCLVResponse(
        customer_id=request.customer_id,
        method=request.method,
        clv=round(clv, 2),
        horizon_months=request.horizon_months
    )


@app.post("/rank_customers_for_retention", response_model=RankCustomersResponse)
async def rank_customers_for_retention(request: RankCustomersRequest):
    """
    Retention campaign prioritization.
    
    Ranks customers for retention targeting based on strategy:
    - **high_clv_high_churn**: High-value customers at risk (CLV × Churn Risk)
    - **high_churn**: Simply highest churn probability
    - **low_p_alive**: Lowest probability of being alive (BG-NBD)
    
    Returns top K customers with their scores.
    """
    engine = get_scoring_engine()
    
    ranked_customers = engine.rank_customers_for_retention(
        request.top_k,
        request.strategy
    )
    
    if not ranked_customers:
        raise HTTPException(
            status_code=500,
            detail="Failed to rank customers"
        )
    
    # Get total customer count
    total_customers = 0
    if engine.loader.customer_data is not None:
        total_customers = len(engine.loader.customer_data)
    
    return RankCustomersResponse(
        strategy=request.strategy,
        total_customers=total_customers,
        customers=ranked_customers
    )


# ============================================================================
# Run Server
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.
    
    Parameters:
    -----------
    host : str
        Host to bind to
    port : int
        Port to bind to
    reload : bool
        Enable auto-reload for development
    """
    uvicorn.run(
        "src.api_service:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()
