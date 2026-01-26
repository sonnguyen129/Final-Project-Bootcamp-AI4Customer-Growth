"""
Model Loader Module

This module handles loading of all trained models from disk for production use.
Models include:
- Classification model (churn prediction)
- BG-NBD model (probability alive, expected transactions)
- Gamma-Gamma model (expected monetary value)
- Cox PH model (survival analysis)
- Preprocessing pipeline (feature scaling)
"""

import os
import pickle
import dill  # For pickling lambda functions
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class ModelLoader:
    """
    Centralized model loading and management class.
    
    Handles loading and caching of all trained models for production predictions.
    """
    
    DEFAULT_MODEL_DIR = "models"
    
    # Model file names
    MODEL_FILES = {
        "classification": "churn_classification_model.pkl",
        "calibrated": "churn_calibrated_model.pkl",
        "bgf": "bgnbd_model.pkl",
        "ggf": "gamma_gamma_model.pkl",
        "cph": "cox_ph_model.pkl",
        "preprocessing": "preprocessing_pipeline.pkl",
        "scaler": "feature_scaler.pkl",
        "feature_columns": "feature_columns.pkl",
        "customer_data": "customer_data.pkl",
        "rfm_bgf": "rfm_bgf_data.pkl",
        "cox_features": "cox_features.pkl",
    }
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize model loader.
        
        Parameters:
        -----------
        model_dir : str, optional
            Directory containing saved models. Defaults to 'models' in project root.
        """
        if model_dir is None:
            # Try to find models directory relative to this file
            current_dir = Path(__file__).parent
            model_dir = current_dir.parent / self.DEFAULT_MODEL_DIR
        
        self.model_dir = Path(model_dir)
        self._models: Dict[str, Any] = {}
        self._loaded = False
        
    def _get_model_path(self, model_name: str) -> Path:
        """Get full path for a model file."""
        if model_name not in self.MODEL_FILES:
            raise ValueError(f"Unknown model: {model_name}")
        return self.model_dir / self.MODEL_FILES[model_name]
    
    def load_model(self, model_name: str, required: bool = True) -> Optional[Any]:
        """
        Load a single model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
        required : bool
            If True, raises error if model not found
            
        Returns:
        --------
        Loaded model object or None if not found and not required
        """
        if model_name in self._models:
            return self._models[model_name]
        
        model_path = self._get_model_path(model_name)
        
        if not model_path.exists():
            if required:
                raise FileNotFoundError(
                    f"Required model '{model_name}' not found at {model_path}. "
                    f"Please run the notebook to train and save models first."
                )
            return None
        
        # Use dill for lifetimes models (bgf, ggf), pickle for others
        if model_name in ["bgf", "ggf"]:
            with open(model_path, 'rb') as f:
                model = dill.load(f)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        self._models[model_name] = model
        return model
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        Load all available models.
        
        Returns:
        --------
        Dictionary of loaded models
        """
        if self._loaded:
            return self._models
        
        print(f"📂 Loading models from: {self.model_dir}")
        
        # Required models - removed customer_data as it will be loaded from CSV
        required_models = ["bgf", "ggf", "cph", "preprocessing", "feature_columns"]
        
        # Optional models - customer_data and rfm_bgf are now optional (for backward compatibility)
        optional_models = ["classification", "calibrated", "scaler", "cox_features", "customer_data", "rfm_bgf"]
        
        for model_name in required_models:
            try:
                self.load_model(model_name, required=True)
                print(f"  ✅ Loaded: {model_name}")
            except FileNotFoundError as e:
                print(f"  ❌ Missing: {model_name}")
                raise e
        
        for model_name in optional_models:
            try:
                model = self.load_model(model_name, required=False)
                if model is not None:
                    print(f"  ✅ Loaded: {model_name}")
                else:
                    print(f"  ⚠️ Optional model not found: {model_name}")
            except Exception as e:
                print(f"  ⚠️ Could not load optional model {model_name}: {e}")
        
        self._loaded = True
        return self._models
    
    @property
    def classification_model(self):
        """Get classification model (calibrated if available, otherwise base model)."""
        calibrated = self._models.get("calibrated")
        if calibrated is not None:
            return calibrated
        return self._models.get("classification")
    
    @property
    def bgf_model(self):
        """Get BG-NBD (BetaGeoFitter) model."""
        return self._models.get("bgf")
    
    @property
    def ggf_model(self):
        """Get Gamma-Gamma (GammaGammaFitter) model."""
        return self._models.get("ggf")
    
    @property
    def cph_model(self):
        """Get Cox Proportional Hazards model."""
        return self._models.get("cph")
    
    @property
    def preprocessing_pipeline(self):
        """Get preprocessing pipeline (imputer + scaler)."""
        return self._models.get("preprocessing")
    
    @property
    def feature_columns(self):
        """Get list of feature columns used for classification model."""
        return self._models.get("feature_columns")
    
    @property
    def customer_data(self) -> Optional[pd.DataFrame]:
        """Get customer feature data."""
        return self._models.get("customer_data")
    
    @property
    def rfm_bgf_data(self) -> Optional[pd.DataFrame]:
        """Get RFM data used for BG-NBD predictions."""
        return self._models.get("rfm_bgf")
    
    @property
    def cox_features(self):
        """Get list of features used in Cox PH model."""
        return self._models.get("cox_features")
    
    # Note: get_customer_features, get_rfm_features, and customer_exists
    # are now handled by DataProcessor class instead of ModelLoader.
    # This allows on-the-fly feature generation from raw CSV data.


def save_models(
    model_dir: str,
    classification_model=None,
    calibrated_model=None,
    bgf_model=None,
    ggf_model=None,
    cph_model=None,
    preprocessing_pipeline=None,
    scaler=None,
    feature_columns=None,
    customer_data=None,
    rfm_bgf_data=None,
    cox_features=None,
):
    """
    Save all models to disk.
    
    This is a utility function to be called from the notebook after training.
    
    Parameters:
    -----------
    model_dir : str
        Directory to save models
    classification_model : sklearn model
        Trained classification model
    calibrated_model : sklearn CalibratedClassifierCV
        Calibrated classification model
    bgf_model : BetaGeoFitter
        Trained BG-NBD model
    ggf_model : GammaGammaFitter
        Trained Gamma-Gamma model
    cph_model : CoxPHFitter
        Trained Cox PH model
    preprocessing_pipeline : sklearn Pipeline
        Fitted preprocessing pipeline
    scaler : StandardScaler
        Fitted feature scaler
    feature_columns : list
        List of feature column names
    customer_data : pd.DataFrame
        Customer feature data for predictions
    rfm_bgf_data : pd.DataFrame
        RFM data for BG-NBD predictions
    cox_features : list
        List of features used in Cox PH model
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_save = {
        "classification": classification_model,
        "calibrated": calibrated_model,
        "bgf": bgf_model,
        "ggf": ggf_model,
        "cph": cph_model,
        "preprocessing": preprocessing_pipeline,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "customer_data": customer_data,
        "rfm_bgf": rfm_bgf_data,
        "cox_features": cox_features,
    }
    
    # Models that require dill (contain lambda functions)
    dill_models = {"bgf", "ggf"}
    
    print(f"📂 Saving models to: {model_dir}")
    
    for name, model in models_to_save.items():
        if model is not None:
            file_name = ModelLoader.MODEL_FILES[name]
            file_path = model_dir / file_name
            
            # Use dill for lifetimes models (contain lambda functions)
            if name in dill_models:
                with open(file_path, 'wb') as f:
                    dill.dump(model, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            print(f"  ✅ Saved: {name} -> {file_name}")
        else:
            print(f"  ⚠️ Skipped: {name} (not provided)")
    
    print("\n✅ Model saving complete!")


# Singleton instance for global access
_loader_instance: Optional[ModelLoader] = None


def get_model_loader(model_dir: Optional[str] = None) -> ModelLoader:
    """
    Get or create the global ModelLoader instance.
    
    Parameters:
    -----------
    model_dir : str, optional
        Model directory path
        
    Returns:
    --------
    ModelLoader instance
    """
    global _loader_instance
    
    if _loader_instance is None:
        _loader_instance = ModelLoader(model_dir)
        _loader_instance.load_all_models()
    
    return _loader_instance


def reset_model_loader():
    """Reset the global ModelLoader instance."""
    global _loader_instance
    _loader_instance = None
