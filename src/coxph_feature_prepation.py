"""
Cox Proportional Hazards Feature Preparation Module
====================================================
This module provides functions to prepare features for Cox PH survival analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def prepare_survival_data(
    features_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    observation_date: pd.Timestamp,
    customer_id_col: str = 'customer_id',
    true_lifetime_col: str = 'true_lifetime_days',
    signup_date_col: str = 'signup_date'
) -> pd.DataFrame:
    """
    Prepare survival data for Cox Proportional Hazards model.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing customer features (e.g., RFM features)
    customers_df : pd.DataFrame
        DataFrame containing customer info with true_lifetime_days and signup_date
    observation_date : pd.Timestamp
        The observation/cutoff date for calculating censor time
    customer_id_col : str
        Column name for customer ID
    true_lifetime_col : str
        Column name for true lifetime days (time to event)
    signup_date_col : str
        Column name for signup date
        
    Returns
    -------
    pd.DataFrame
        DataFrame with survival analysis columns: T, censor_time, duration, event
    """
    # Copy features dataframe
    survival_df = features_df.copy()
    
    # Ensure signup_date is datetime
    customers_df = customers_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(customers_df[signup_date_col]):
        customers_df[signup_date_col] = pd.to_datetime(customers_df[signup_date_col])
    
    # Merge with customers data
    survival_df = survival_df.merge(
        customers_df[[customer_id_col, true_lifetime_col, signup_date_col]],
        on=customer_id_col,
        how='left'
    )
    
    # Calculate censor_time: difference between observation date and signup date
    survival_df['censor_time'] = (observation_date - survival_df[signup_date_col]).dt.days
    
    # T = true_lifetime_days (time to event - actual churn time)
    survival_df['T'] = survival_df[true_lifetime_col]
    
    # Duration: min(T, censor_time) - the observed time
    survival_df['duration'] = np.minimum(survival_df['T'], survival_df['censor_time'])
    survival_df['duration'] = survival_df['duration'].clip(lower=1)  # Minimum 1 day
    
    # Event: 1 if T <= censor_time (event occurred before censoring), 0 otherwise (censored)
    survival_df['event'] = (survival_df['T'] <= survival_df['censor_time']).astype(int)
    
    return survival_df


def prepare_cox_features(
    survival_df: pd.DataFrame,
    feature_cols: List[str],
    duration_col: str = 'duration',
    event_col: str = 'event',
    fillna_value: float = 0
) -> pd.DataFrame:
    """
    Prepare feature matrix for Cox PH model fitting.
    
    Parameters
    ----------
    survival_df : pd.DataFrame
        DataFrame with survival data
    feature_cols : List[str]
        List of feature column names to include
    duration_col : str
        Column name for duration
    event_col : str
        Column name for event indicator
    fillna_value : float
        Value to fill NaN with
        
    Returns
    -------
    pd.DataFrame
        DataFrame ready for CoxPHFitter
    """
    required_cols = feature_cols + [duration_col, event_col]
    cox_df = survival_df[required_cols].copy()
    cox_df = cox_df.fillna(fillna_value)
    
    return cox_df


def get_default_cox_features() -> List[str]:
    """
    Get the default list of features for Cox PH model.
    
    Returns
    -------
    List[str]
        Default feature column names
    """
    return [
        'recency', 'frequency', 'monetary', 'RFM_score',
        'purchase_frequency_rate', 'recent_30d_frequency', 'frequency_trend',
        'is_declining', 'cv_spending'
    ]


def print_survival_summary(survival_df: pd.DataFrame, observation_date: pd.Timestamp) -> None:
    """
    Print summary statistics of survival data.
    
    Parameters
    ----------
    survival_df : pd.DataFrame
        DataFrame with survival data
    observation_date : pd.Timestamp
        The observation date used
    """
    print("=" * 60)
    print("SURVIVAL DATA SUMMARY")
    print("=" * 60)
    print(f"\nObservation Date: {observation_date.strftime('%Y-%m-%d')}")
    print(f"\nDataset Size: {survival_df.shape[0]} customers")
    print(f"   - Events (T <= censor_time): {survival_df['event'].sum()}")
    print(f"   - Censored (T > censor_time): {(1 - survival_df['event']).sum()}")
    print(f"   - Event Rate: {survival_df['event'].mean():.2%}")
    
    print(f"\nDuration Statistics:")
    print(f"   - Mean: {survival_df['duration'].mean():.1f} days")
    print(f"   - Median: {survival_df['duration'].median():.1f} days")
    print(f"   - Min: {survival_df['duration'].min():.1f} days")
    print(f"   - Max: {survival_df['duration'].max():.1f} days")
    
    print(f"\nT (True Lifetime) Statistics:")
    print(f"   - Mean: {survival_df['T'].mean():.1f} days")
    print(f"   - Median: {survival_df['T'].median():.1f} days")
    
    print(f"\nCensor Time Statistics:")
    print(f"   - Mean: {survival_df['censor_time'].mean():.1f} days")
    print(f"   - Median: {survival_df['censor_time'].median():.1f} days")
