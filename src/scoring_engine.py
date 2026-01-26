"""
Scoring Engine Module

This module contains the core business logic for customer scoring, churn prediction,
survival analysis, and CLV estimation.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .model_loader import ModelLoader, get_model_loader
from .data_processor import DataProcessor, get_data_processor
from .models import (
    ChurnRiskLabel, CLVMethod, RetentionStrategy,
    SurvivalPoint, RankedCustomer
)


@dataclass
class CustomerScore:
    """Complete customer score data"""
    customer_id: str
    churn_probability: float
    p_alive: float
    expected_remaining_lifetime: float
    clv_bgnbd: float
    clv_survival: float


class ScoringEngine:
    """
    Core scoring engine for customer predictions.
    
    This class provides all prediction functionality using trained models.
    """
    
    # Default parameters
    DISCOUNT_RATE = 0.01  # Monthly discount rate
    DEFAULT_HORIZON_MONTHS = 12
    SURVIVAL_TIME_POINTS = [30, 60, 90, 180, 365]
    
    def __init__(self, model_loader: Optional[ModelLoader] = None, data_processor: Optional[DataProcessor] = None):
        """
        Initialize scoring engine.
        
        Parameters:
        -----------
        model_loader : ModelLoader, optional
            Pre-configured model loader. If None, uses global instance.
        data_processor : DataProcessor, optional
            Pre-configured data processor. If None, uses global instance.
        """
        self.loader = model_loader or get_model_loader()
        self.processor = data_processor or get_data_processor()
        
    def _get_churn_label(self, probability: float) -> ChurnRiskLabel:
        """Convert churn probability to risk label."""
        if probability < 0.3:
            return ChurnRiskLabel.LOW_RISK
        elif probability < 0.5:
            return ChurnRiskLabel.MEDIUM_RISK
        elif probability < 0.7:
            return ChurnRiskLabel.HIGH_RISK
        else:
            return ChurnRiskLabel.CRITICAL_RISK
    
    def _calculate_bgnbd_clv(
        self, 
        frequency: float, 
        recency: float, 
        T: float, 
        monetary_value: float,
        horizon_months: int = 12
    ) -> float:
        """
        Calculate CLV using BG-NBD + Gamma-Gamma method.
        
        Parameters:
        -----------
        frequency : float
            Number of repeat purchases
        recency : float
            Days from first to last purchase
        T : float
            Customer age in days
        monetary_value : float
            Average transaction value
        horizon_months : int
            CLV prediction horizon
            
        Returns:
        --------
        CLV value
        """
        bgf = self.loader.bgf_model
        ggf = self.loader.ggf_model
        
        if bgf is None or ggf is None:
            return monetary_value * horizon_months * 0.5  # Fallback
        
        try:
            clv = ggf.customer_lifetime_value(
                bgf,
                frequency,
                recency,
                T,
                monetary_value,
                time=horizon_months,
                discount_rate=self.DISCOUNT_RATE
            )
            return float(clv) if not np.isnan(clv) else 0.0
        except Exception:
            return monetary_value * horizon_months * 0.5
    
    def _calculate_survival_clv(
        self,
        customer_features: pd.DataFrame,
        expected_profit: float,
        monthly_purchases: float,
        horizon_months: int = 12
    ) -> float:
        """
        Calculate CLV using survival analysis approach.
        
        Parameters:
        -----------
        customer_features : pd.DataFrame
            Customer features for Cox PH model
        expected_profit : float
            Expected profit per transaction
        monthly_purchases : float
            Expected monthly purchase rate
        horizon_months : int
            CLV prediction horizon
            
        Returns:
        --------
        CLV value
        """
        cph = self.loader.cph_model
        
        if cph is None:
            return expected_profit * monthly_purchases * horizon_months * 0.5
        
        try:
            # Get survival probabilities at different time points
            horizon_days = horizon_months * 30
            time_points = [t for t in self.SURVIVAL_TIME_POINTS if t <= horizon_days]
            if not time_points:
                time_points = [horizon_days]
            
            surv_probs = cph.predict_survival_function(customer_features, times=time_points)
            avg_survival = surv_probs.values.mean()
            
            # Discount factor approximation
            discount_factor = (1 - (1 + self.DISCOUNT_RATE) ** (-horizon_months)) / self.DISCOUNT_RATE
            
            clv = expected_profit * monthly_purchases * avg_survival * discount_factor
            return max(0.0, float(clv))
        except Exception:
            return expected_profit * monthly_purchases * horizon_months * 0.5
    
    def score_customer(self, customer_id: str) -> Optional[CustomerScore]:
        """
        Generate unified customer score.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID to score
            
        Returns:
        --------
        CustomerScore object or None if customer not found
        """
        # Check if customer exists
        if not self.processor.customer_exists(customer_id):
            return None
        
        # Compute features on-the-fly
        customer_features = self.processor.compute_all_features(customer_id)
        if customer_features is None:
            return None
        
        # Get RFM features
        rfm_features_dict = self.processor.compute_rfm_features(customer_id)
        if rfm_features_dict is None:
            rfm_features = None
        else:
            rfm_features = pd.Series(rfm_features_dict)
        
        # 1. Classification-based churn probability
        churn_prob = self._predict_churn_classification(customer_features)
        
        # 2. BG-NBD P(alive)
        p_alive = self._predict_p_alive(rfm_features)
        
        # 3. Expected remaining lifetime (from Cox PH)
        expected_lifetime = self._predict_remaining_lifetime(customer_features)
        
        # 4. CLV calculations
        clv_bgnbd = self._calculate_customer_clv_bgnbd(rfm_features)
        clv_survival = self._calculate_customer_clv_survival(customer_features, rfm_features)
        
        return CustomerScore(
            customer_id=customer_id,
            churn_probability=churn_prob,
            p_alive=p_alive,
            expected_remaining_lifetime=expected_lifetime,
            clv_bgnbd=clv_bgnbd,
            clv_survival=clv_survival
        )
    
    def _predict_churn_classification(self, customer_features: pd.DataFrame) -> float:
        """Predict churn probability using classification model."""
        model = self.loader.classification_model
        feature_cols = self.loader.feature_columns
        preprocessing = self.loader.preprocessing_pipeline
        
        if model is None:
            # Fallback: use 1 - p_alive if available
            return 0.5
        
        try:
            # Ensure feature_cols are available
            available_cols = [c for c in feature_cols if c in customer_features.columns]
            if len(available_cols) == 0:
                return 0.5
            
            # Prepare features - make a deep copy and ensure clean state
            X = customer_features[available_cols].copy(deep=True)
            X = X.reset_index(drop=True)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Apply preprocessing - use .values to avoid index issues with sklearn
            if preprocessing is not None:
                X_scaled = preprocessing.transform(X.values)
            else:
                X_scaled = X.values
            
            # Predict
            prob = model.predict_proba(X_scaled)[:, 1]
            return float(prob[0])
        except Exception as e:
            print(f"Warning: Classification prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5
    
    def _predict_p_alive(self, rfm_features: Optional[pd.Series]) -> float:
        """Predict probability of being alive using BG-NBD."""
        bgf = self.loader.bgf_model
        
        if bgf is None or rfm_features is None:
            return 0.5
        
        try:
            p_alive = bgf.conditional_probability_alive(
                rfm_features['frequency'],
                rfm_features['recency'],
                rfm_features['T']
            )
            return float(p_alive)
        except Exception:
            return 0.5
    
    def _predict_remaining_lifetime(self, customer_features: pd.DataFrame) -> float:
        """
        Predict expected remaining lifetime using Cox PH.
        
        Calculation (from notebook):
        - median_survival_time = cph.predict_median(cox_features)
        - expected_remaining_lifetime = median_survival_time - duration
        - expected_remaining_lifetime = clip(lower=0)
        
        Returns:
        --------
        Expected remaining lifetime in days (clipped at 0)
        """
        cph = self.loader.cph_model
        cox_features = self.loader.cox_features
        
        if cph is None or cox_features is None:
            return 180.0  # Default 180 days (~6 months)
        
        try:
            # Get Cox features - use deep copy and reset index for consistent state
            available_features = [f for f in cox_features if f in customer_features.columns]
            X_cox = customer_features[available_features].copy(deep=True)
            X_cox = X_cox.reset_index(drop=True)
            X_cox = X_cox.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Predict median survival time (in days)
            median_survival = cph.predict_median(X_cox)
            
            # Get customer's current duration
            duration = customer_features['duration'].values[0] if 'duration' in customer_features.columns else 0
            
            # Calculate expected remaining lifetime (in days)
            # Formula: expected_remaining_lifetime = median_survival_time - duration
            median_survival_value = float(median_survival.values[0])
            expected_remaining_lifetime = median_survival_value - duration
            
            # Clip at 0 (cannot have negative remaining lifetime)
            return max(0.0, expected_remaining_lifetime)
        except Exception:
            return 180.0  # Default 180 days
    
    def _calculate_customer_clv_bgnbd(self, rfm_features: Optional[pd.Series]) -> float:
        """Calculate CLV using BG-NBD + Gamma-Gamma for a customer."""
        if rfm_features is None:
            return 0.0
        
        return self._calculate_bgnbd_clv(
            rfm_features['frequency'],
            rfm_features['recency'],
            rfm_features['T'],
            rfm_features.get('monetary_value', 0),
            self.DEFAULT_HORIZON_MONTHS
        )
    
    def _calculate_customer_clv_survival(
        self, 
        customer_features: pd.DataFrame,
        rfm_features: Optional[pd.Series]
    ) -> float:
        """Calculate CLV using survival approach for a customer."""
        cph = self.loader.cph_model
        ggf = self.loader.ggf_model
        cox_features = self.loader.cox_features
        
        if rfm_features is None:
            return 0.0
        
        # Get expected profit
        if ggf is not None and rfm_features['frequency'] > 0:
            try:
                expected_profit = ggf.conditional_expected_average_profit(
                    rfm_features['frequency'],
                    rfm_features['monetary_value']
                )
                expected_profit = float(expected_profit)
            except Exception:
                expected_profit = rfm_features.get('monetary_value', 0)
        else:
            expected_profit = rfm_features.get('monetary_value', 0)
        
        # Estimate monthly purchases
        T = rfm_features['T']
        frequency = rfm_features['frequency']
        monthly_purchases = (frequency + 1) / (T / 30) if T > 0 else 0.5
        
        # Prepare Cox features - use deep copy and reset index
        if cph is not None and cox_features is not None:
            available_features = [f for f in cox_features if f in customer_features.columns]
            X_cox = customer_features[available_features].copy(deep=True)
            X_cox = X_cox.reset_index(drop=True)
            X_cox = X_cox.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            X_cox = None
        
        return self._calculate_survival_clv(
            X_cox if X_cox is not None else pd.DataFrame(),
            expected_profit,
            monthly_purchases,
            self.DEFAULT_HORIZON_MONTHS
        )
    
    def predict_churn(
        self, 
        customer_id: str, 
        horizon_days: int = 60
    ) -> Optional[Tuple[float, ChurnRiskLabel]]:
        """
        Predict churn probability and risk label.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
        horizon_days : int
            Prediction horizon in days
            
        Returns:
        --------
        Tuple of (probability, label) or None if customer not found
        """
        # Compute features on-the-fly
        customer_features = self.processor.compute_all_features(customer_id)
        if customer_features is None:
            return None
        
        churn_prob = self._predict_churn_classification(customer_features)
        
        # Adjust probability based on horizon (rough adjustment)
        if horizon_days != 60:
            adjustment_factor = horizon_days / 60
            churn_prob = min(1.0, churn_prob * adjustment_factor)
        
        label = self._get_churn_label(churn_prob)
        
        return churn_prob, label
    
    def predict_survival_curve(
        self, 
        customer_id: str
    ) -> Optional[Tuple[List[SurvivalPoint], float]]:
        """
        Predict survival curve and expected remaining lifetime.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
            
        Returns:
        --------
        Tuple of (survival_curve, expected_remaining_lifetime) or None
        """
        # Compute features on-the-fly
        customer_features = self.processor.compute_all_features(customer_id)
        if customer_features is None:
            return None
        
        cph = self.loader.cph_model
        cox_features = self.loader.cox_features
        
        if cph is None or cox_features is None:
            # Fallback: generate default curve
            default_curve = [
                SurvivalPoint(day=30, prob=0.85),
                SurvivalPoint(day=60, prob=0.70),
                SurvivalPoint(day=90, prob=0.55),
            ]
            return default_curve, 6.0
        
        try:
            # Prepare features - use deep copy and reset index
            available_features = [f for f in cox_features if f in customer_features.columns]
            X_cox = customer_features[available_features].copy(deep=True)
            X_cox = X_cox.reset_index(drop=True)
            X_cox = X_cox.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Generate survival curve
            time_points = [30, 60, 90, 180, 365]
            surv_func = cph.predict_survival_function(X_cox, times=time_points)
            
            survival_curve = [
                SurvivalPoint(day=t, prob=float(surv_func.iloc[i, 0]))
                for i, t in enumerate(time_points)
            ]
            
            # Expected remaining lifetime
            expected_lifetime = self._predict_remaining_lifetime(customer_features)
            
            return survival_curve, expected_lifetime
        except Exception as e:
            print(f"Warning: Survival prediction failed: {e}")
            default_curve = [
                SurvivalPoint(day=30, prob=0.85),
                SurvivalPoint(day=60, prob=0.70),
                SurvivalPoint(day=90, prob=0.55),
            ]
            return default_curve, 6.0
    
    def estimate_clv(
        self, 
        customer_id: str, 
        method: CLVMethod = CLVMethod.BGNBD,
        horizon_months: int = 12
    ) -> Optional[float]:
        """
        Estimate customer lifetime value.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
        method : CLVMethod
            CLV calculation method
        horizon_months : int
            Prediction horizon in months
            
        Returns:
        --------
        CLV value or None if customer not found
        """
        # Compute features on-the-fly
        customer_features = self.processor.compute_all_features(customer_id)
        if customer_features is None:
            return None
        
        rfm_features_dict = self.processor.compute_rfm_features(customer_id)
        if rfm_features_dict is None:
            rfm_features = None
        else:
            rfm_features = pd.Series(rfm_features_dict)
        
        if method == CLVMethod.BGNBD:
            if rfm_features is None:
                return 0.0
            return self._calculate_bgnbd_clv(
                rfm_features['frequency'],
                rfm_features['recency'],
                rfm_features['T'],
                rfm_features.get('monetary_value', 0),
                horizon_months
            )
        else:  # Survival method
            return self._calculate_customer_clv_survival(customer_features, rfm_features)
    
    def rank_customers_for_retention(
        self,
        top_k: int = 100,
        strategy: RetentionStrategy = RetentionStrategy.HIGH_CLV_HIGH_CHURN
    ) -> List[RankedCustomer]:
        """
        Rank customers for retention prioritization.
        
        Parameters:
        -----------
        top_k : int
            Number of customers to return
        strategy : RetentionStrategy
            Ranking strategy
            
        Returns:
        --------
        List of ranked customers
        """
        # Get all customer IDs
        all_customer_ids = self.processor.get_all_customer_ids()
        
        if len(all_customer_ids) == 0:
            return []
        
        # Compute features for all customers (this might be slow for large datasets)
        # In production, consider batching or caching
        print(f"Computing features for {len(all_customer_ids)} customers...")
        
        customer_scores = []
        for i, customer_id in enumerate(all_customer_ids):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(all_customer_ids)}...")
            
            try:
                customer_features = self.processor.compute_all_features(customer_id)
                if customer_features is None:
                    continue
                
                rfm_dict = self.processor.compute_rfm_features(customer_id)
                if rfm_dict is None:
                    continue
                
                # Predict churn
                churn_prob = self._predict_churn_classification(customer_features)
                
                customer_scores.append({
                    'customer_id': customer_id,
                    'churn_prob': churn_prob,
                    'clv': rfm_dict.get('total_spent', 0)
                })
            except Exception as e:
                print(f"Warning: Failed to process {customer_id}: {e}")
                continue
        
        if len(customer_scores) == 0:
            return []
        
        df = pd.DataFrame(customer_scores)
        
        # Calculate priority score based on strategy
        if strategy == RetentionStrategy.HIGH_CLV_HIGH_CHURN:
            # Normalize and combine
            df['clv_norm'] = (df['clv'] - df['clv'].min()) / (df['clv'].max() - df['clv'].min() + 1e-6)
            df['priority_score'] = df['clv_norm'] * df['churn_prob']
        elif strategy == RetentionStrategy.HIGH_CHURN:
            df['priority_score'] = df['churn_prob']
        else:  # LOW_P_ALIVE
            # For LOW_P_ALIVE strategy, we already have churn_prob which is approximately 1 - p_alive
            df['priority_score'] = df['churn_prob']
        
        # Sort and get top K
        df_sorted = df.nlargest(top_k, 'priority_score')
        
        # Convert to response format
        ranked_customers = [
            RankedCustomer(
                customer_id=row['customer_id'],
                churn_probability=float(row['churn_prob']),
                clv=float(row['clv']),
                priority_score=float(row['priority_score'])
            )
            for _, row in df_sorted.iterrows()
        ]
        
        return ranked_customers


# Singleton instance
_engine_instance: Optional[ScoringEngine] = None


def get_scoring_engine() -> ScoringEngine:
    """Get or create global ScoringEngine instance."""
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = ScoringEngine()
    
    return _engine_instance


def reset_scoring_engine():
    """Reset the global ScoringEngine instance."""
    global _engine_instance
    _engine_instance = None
