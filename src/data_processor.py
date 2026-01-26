"""
Data Processor Module

This module handles loading raw CSV data and processing it on-the-fly
to generate features for customer scoring.

When the API starts, it loads customers.csv and transactions.csv.
When an API request is made, it generates features for that specific customer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

from .feature_engineer import ChurnFeatureEngineer


class DataProcessor:
    """
    Handles loading raw data and generating features on-demand.
    
    This class:
    1. Loads CSV files on initialization
    2. Computes features for specific customers when requested
    3. Supports both RFM-based features and full feature sets
    """
    
    DEFAULT_DATA_DIR = "data"
    
    def __init__(self, data_dir: Optional[str] = None, observation_date: Optional[pd.Timestamp] = None):
        """
        Initialize data processor.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing CSV files. Defaults to 'data' in project root.
        observation_date : pd.Timestamp, optional
            Reference date for feature calculation. Defaults to today.
        """
        if data_dir is None:
            # Try to find data directory relative to this file
            current_dir = Path(__file__).parent
            data_dir = current_dir.parent / self.DEFAULT_DATA_DIR
        
        self.data_dir = Path(data_dir)
        self.observation_date = observation_date or pd.Timestamp.now()
        
        # Data storage
        self.customers: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None
        self.all_features: Optional[pd.DataFrame] = None  # Pre-computed features for all customers
        
        # Feature engineer for generating features
        self.feature_engineer: Optional[ChurnFeatureEngineer] = None
        
        # Cache for computed features (optional optimization)
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        
    def load_data(self) -> None:
        """
        Load customers and transactions from CSV files.
        
        Raises:
        -------
        FileNotFoundError if CSV files are not found
        """
        customers_path = self.data_dir / "customers.csv"
        transactions_path = self.data_dir / "transactions.csv"
        
        if not customers_path.exists():
            raise FileNotFoundError(
                f"customers.csv not found at {customers_path}. "
                f"Please ensure the data files are in the correct location."
            )
        
        if not transactions_path.exists():
            raise FileNotFoundError(
                f"transactions.csv not found at {transactions_path}. "
                f"Please ensure the data files are in the correct location."
            )
        
        print(f"📂 Loading data from: {self.data_dir}")
        
        # Load customers
        self.customers = pd.read_csv(customers_path)
        
        # Convert date columns
        date_cols = ['signup_date', 'last_purchase_date']
        for col in date_cols:
            if col in self.customers.columns:
                self.customers[col] = pd.to_datetime(self.customers[col])
        
        print(f"  ✅ Loaded customers: {len(self.customers):,} records")
        
        # Load transactions
        self.transactions = pd.read_csv(transactions_path)
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        
        print(f"  ✅ Loaded transactions: {len(self.transactions):,} records")
        
        # Initialize feature engineer
        self.feature_engineer = ChurnFeatureEngineer(
            observation_date=self.observation_date,
            horizon=60,
            historical_window=365,  # Use 1 year of history for features
            verbose=False
        )
        
        print(f"  ✅ Feature engineer initialized with observation_date={self.observation_date.date()}")
        
        # Pre-compute features for all customers (this enables RFM scoring to work)
        print(f"\n  🔄 Pre-computing features for all customers...")
        try:
            self.all_features = self.feature_engineer.transform(self.transactions)
            print(f"  ✅ Features computed for {len(self.all_features):,} active customers")
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to pre-compute features: {e}")
            self.all_features = None
    
    def customer_exists(self, customer_id: str) -> bool:
        """Check if a customer exists in the data."""
        if self.customers is None:
            return False
        return customer_id in self.customers['customer_id'].values
    
    def get_customer_info(self, customer_id: str) -> Optional[pd.Series]:
        """
        Get basic customer information.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
            
        Returns:
        --------
        Series with customer info or None if not found
        """
        if self.customers is None:
            return None
        
        customer = self.customers[self.customers['customer_id'] == customer_id]
        if len(customer) == 0:
            return None
        
        return customer.iloc[0]
    
    def get_customer_transactions(self, customer_id: str) -> pd.DataFrame:
        """
        Get all transactions for a specific customer.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
            
        Returns:
        --------
        DataFrame with customer transactions
        """
        if self.transactions is None:
            return pd.DataFrame()
        
        return self.transactions[self.transactions['customer_id'] == customer_id].copy()
    
    def compute_rfm_features(self, customer_id: str) -> Optional[Dict[str, float]]:
        """
        Compute RFM features for a specific customer.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
            
        Returns:
        --------
        Dictionary with RFM features: frequency, recency, T, monetary_value
        """
        customer_txns = self.get_customer_transactions(customer_id)
        
        if len(customer_txns) == 0:
            return None
        
        # Sort by date
        customer_txns = customer_txns.sort_values('transaction_date')
        
        # Get first and last transaction dates
        first_purchase = customer_txns['transaction_date'].min()
        last_purchase = customer_txns['transaction_date'].max()
        
        # Calculate RFM metrics
        # Frequency: number of repeat purchases (total - 1)
        frequency = max(0, len(customer_txns) - 1)
        
        # Recency: days from first to last purchase
        recency = (last_purchase - first_purchase).days
        
        # T: customer age (days from first purchase to observation date)
        T = (self.observation_date - first_purchase).days
        
        # Monetary value: average transaction value (for repeat customers)
        if frequency > 0:
            monetary_value = customer_txns['amount'].mean()
        else:
            monetary_value = customer_txns['amount'].iloc[0]
        
        return {
            'frequency': float(frequency),
            'recency': float(recency),
            'T': float(T),
            'monetary_value': float(monetary_value),
            'total_spent': float(customer_txns['amount'].sum()),
            'transaction_count': int(len(customer_txns))
        }
    
    def compute_all_features(self, customer_id: str) -> Optional[pd.DataFrame]:
        """
        Compute all features for a specific customer using ChurnFeatureEngineer.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
            
        Returns:
        --------
        DataFrame with all features for the customer (single row)
        """
        # If we have pre-computed features, use them
        if self.all_features is not None:
            # Always return a deep copy with reset index to avoid mutation issues
            # and ensure consistent indexing for sklearn operations
            customer_features = self.all_features[self.all_features['customer_id'] == customer_id].copy(deep=True)
            
            if len(customer_features) > 0:
                # Reset index to ensure consistent 0-based indexing
                customer_features = customer_features.reset_index(drop=True)
                return customer_features
        
        # Fallback: customer not found in pre-computed features
        # This might happen nếu customer không có transaction trong observation window
        # Trả về DataFrame với giá trị mặc định để tránh lỗi 500
        import pandas as pd
        default_features = pd.DataFrame({
            'customer_id': [customer_id],
        })
        return default_features
    
    def compute_cox_features(self, customer_id: str, cox_feature_list: list) -> Optional[pd.DataFrame]:
        """
        Compute features for Cox PH model.
        
        Parameters:
        -----------
        customer_id : str
            Customer ID
        cox_feature_list : list
            List of feature names needed for Cox model
            
        Returns:
        --------
        DataFrame with Cox features for the customer
        """
        all_features = self.compute_all_features(customer_id)
        
        if all_features is None:
            return None
        
        # Extract only the features needed for Cox model
        available_features = [f for f in cox_feature_list if f in all_features.columns]
        
        if len(available_features) == 0:
            return None
        
        cox_features = all_features[available_features].copy()
        
        # Handle missing values
        cox_features = cox_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return cox_features
    
    def get_all_customer_ids(self) -> list:
        """Get list of all customer IDs."""
        if self.customers is None:
            return []
        return self.customers['customer_id'].tolist()
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._feature_cache.clear()
    
    def set_observation_date(self, new_date: pd.Timestamp):
        """
        Update the observation date and reinitialize feature engineer.
        
        Parameters:
        -----------
        new_date : pd.Timestamp
            New observation date
        """
        self.observation_date = new_date
        
        # Reinitialize feature engineer with new date
        if self.feature_engineer is not None:
            self.feature_engineer = ChurnFeatureEngineer(
                observation_date=self.observation_date,
                horizon=60,
                historical_window=365,
                verbose=False
            )
        
        # Clear cache as features depend on observation date
        self.clear_cache()
        
        print(f"✅ Observation date updated to {self.observation_date.date()}")


# Singleton instance for global access
_processor_instance: Optional[DataProcessor] = None


def get_data_processor(data_dir: Optional[str] = None, observation_date: Optional[pd.Timestamp] = None) -> DataProcessor:
    """
    Get or create the global DataProcessor instance.
    
    Parameters:
    -----------
    data_dir : str, optional
        Data directory path
    observation_date : pd.Timestamp, optional
        Reference date for features
        
    Returns:
    --------
    DataProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = DataProcessor(data_dir, observation_date)
        _processor_instance.load_data()
    
    return _processor_instance


def reset_data_processor():
    """Reset the global DataProcessor instance."""
    global _processor_instance
    _processor_instance = None
