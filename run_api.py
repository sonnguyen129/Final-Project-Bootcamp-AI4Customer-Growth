"""
Main entry point for running the Customer Churn & CLV API

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Example:
    python run_api.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
from pathlib import Path
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run the Customer Churn & CLV API Server"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Customer Churn & CLV API Server")
    print("=" * 60)
    print(f"\n📍 Server URL: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"📋 OpenAPI: http://{args.host}:{args.port}/openapi.json")
    print("\n" + "=" * 60)
    print("\n📌 Available Endpoints:")
    print("  POST /score_customer - Unified customer scoring")
    print("  POST /predict_churn - Churn prediction")
    print("  POST /predict_survival - Survival analysis")
    print("  POST /estimate_clv - CLV estimation")
    print("  POST /rank_customers_for_retention - Retention ranking")
    print("  GET  /health - Health check")
    print("\n" + "=" * 60 + "\n")
    
    # Run server using uvicorn
    uvicorn.run(
        "src.api_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
