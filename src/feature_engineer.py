"""
Feature Engineering Pipeline for Customer Churn Prediction

This module contains the ChurnFeatureEngineer class that provides a unified
pipeline for calculating features from transaction data for both train and test sets.

Author: Data Science Team
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Pipeline để tính toán features từ transaction data.
    Đảm bảo tính nhất quán cho cả train và test set.
    
    Features được tạo ra bao gồm:
    - RFM Features (Recency, Frequency, Monetary)
    - Transaction Statistics
    - Time-based Trend Features
    - Purchase Interval Features
    - RFM Scores và Segmentation
    - Churn Labels
    
    Example:
    --------
    >>> pipeline = ChurnFeatureEngineer(
    ...     observation_date=pd.Timestamp('2025-10-01'),
    ...     horizon=60,
    ...     historical_window=90
    ... )
    >>> features = pipeline.fit_transform(transactions_df)
    """
    
    # Feature columns list for model training
    FEATURE_COLUMNS = [
        # RFM Features
        'recency', 'frequency', 'monetary', 'total_spent',
        'customer_age_days', 'active_duration',
        
        # Transaction Statistics
        'avg_transaction', 'std_transaction', 'median_transaction',
        'purchase_frequency_rate', 'avg_days_between_purchases',
        'cv_spending', 'spending_per_day',
        
        # Trend Features  
        'recent_30d_frequency', 'recent_30d_amount', 'recent_30d_avg',
        'previous_30d_frequency', 'previous_30d_amount', 'previous_30d_avg',
        'last_90d_frequency', 'last_90d_amount',
        'frequency_trend', 'amount_trend',
        'frequency_trend_ratio', 'amount_trend_ratio',
        'is_frequency_declining', 'is_amount_declining', 'is_declining',
        'no_activity_30d', 'no_activity_60d',
        
        # Interval Features
        'mean_interval', 'std_interval', 'min_interval', 'max_interval',
        'last_interval', 'cv_interval', 'recency_vs_mean_interval',
        
        # RFM Scores
        'R_score', 'F_score', 'M_score', 'RFM_score',
        'is_high_value', 'is_at_risk',
        
        # Advanced Relative Features
        'recency_to_tenure_ratio', 'recent_frequency_ratio', 'recent_amount_ratio',
        'recent_avg_vs_overall', 'RFM_score_normalized'
    ]
    
    def __init__(self, observation_date, horizon=60, historical_window=90, verbose=True):
        """
        Parameters:
        -----------
        observation_date : pd.Timestamp
            Ngày quan sát (cutoff date cho tập dữ liệu)
        horizon : int
            Cửa sổ dự báo (default: 60 ngày)
            - Dùng để lọc KH active: recency < horizon
            - Dùng để tính label: KH có mua trong horizon ngày tới không?
        historical_window : int
            Số ngày quá khứ để tính features (default: 90)
            Chỉ lấy transactions từ (observation_date - historical_window) đến observation_date
        verbose : bool
            In thông tin trong quá trình xử lý (default: True)
        """
        self.observation_date = pd.to_datetime(observation_date)
        self.horizon = horizon
        self.historical_window = historical_window
        self.verbose = verbose
        
    def fit(self, transactions_df, y=None):
        """Fit không làm gì - chỉ để tuân theo sklearn API"""
        return self
    
    def transform(self, transactions_df):
        """
        Tính toán tất cả features từ transaction data
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            DataFrame chứa transactions với columns: customer_id, transaction_date, amount
        
        Returns:
        --------
        pd.DataFrame: DataFrame với đầy đủ features cho từng customer
        """
        if self.verbose:
            print(f"\n🔄 Đang xử lý features với observation_date = {self.observation_date.strftime('%d/%m/%Y')}...")
        
        # ==============================================================================
        # CRITICAL FIX: Tách dữ liệu quá khứ và tương lai
        # ==============================================================================
        # 1. Dữ liệu quá khứ: để tính features 
        #    Chỉ lấy transactions từ (observation_date - historical_window) đến observation_date
        historical_start_date = self.observation_date - pd.Timedelta(days=self.historical_window)
        historical_data = transactions_df[
            (transactions_df['transaction_date'] >= historical_start_date) &
            (transactions_df['transaction_date'] < self.observation_date)
        ].copy()
        
        # 2. Dữ liệu tương lai: để tính labels (transactions trong horizon)
        future_end_date = self.observation_date + pd.Timedelta(days=self.horizon)
        future_data = transactions_df[
            (transactions_df['transaction_date'] >= self.observation_date) & 
            (transactions_df['transaction_date'] < future_end_date)
        ].copy()
        
        if historical_data.empty:
            raise ValueError(f"Không có dữ liệu historical trong khoảng {historical_start_date} - {self.observation_date}")
        
        if self.verbose:
            print(f"  📊 Historical window: {self.historical_window} days ({historical_start_date.strftime('%Y-%m-%d')} to {self.observation_date.strftime('%Y-%m-%d')})")
            print(f"  📊 Historical transactions: {len(historical_data):,}")
            print(f"  📊 Future transactions (for labeling): {len(future_data):,}")
        
        # ==============================================================================
        # Tính toán features từ dữ liệu historical
        # ==============================================================================
        # 1. Basic RFM Features
        features = self._calculate_rfm_features(historical_data)
        
        # 2. Transaction Statistics Features
        features = self._add_transaction_stats(features, historical_data)
        
        # 3. Time-based Trend Features
        features = self._add_trend_features(features, historical_data)
        
        # 4. Purchase Interval Features
        features = self._add_interval_features(features, historical_data)
        
        # 5. RFM Segmentation Scores
        features = self._add_rfm_scores(features)
        
        # ==============================================================================
        # CRITICAL FIX: FILTERING - Loại bỏ khách hàng đã churn
        # Chỉ giữ lại những KH đang "active" tại thời điểm observation_date
        # ==============================================================================
        original_count = len(features)
        features = features[features['recency'] < self.horizon].copy()
        
        if self.verbose:
            print(f"\n  🔍 FILTERING STEP:")
            print(f"     Original customers: {original_count:,}")
            print(f"     Active customers (recency < {self.horizon}): {len(features):,}")
            print(f"     Filtered out (already churned): {original_count - len(features):,}")
        
        # ==============================================================================
        # CRITICAL FIX: Label calculation - Dựa trên hành vi tương lai
        # ==============================================================================
        # Label = 1 (Churn) nếu KHÔNG có giao dịch nào trong prediction_window
        # Label = 0 (Active/Retained) nếu CÓ giao dịch trong prediction_window
        buyers_in_future = future_data['customer_id'].unique()
        
        # Tạo cột label
        features['label'] = 1  # Default: Churn
        # Những người có mua trong future => label = 0 (Retained)
        features.loc[features['customer_id'].isin(buyers_in_future), 'label'] = 0
        
        # Thêm cột text label để dễ hiểu
        features['churn_label'] = features['label'].map({0: 'Retained', 1: 'Churned'})
        
        if self.verbose:
            churn_count = features['label'].sum()
            retained_count = len(features) - churn_count
            print(f"\n  📊 LABEL DISTRIBUTION:")
            print(f"     Retained (label=0): {retained_count:,} ({retained_count/len(features)*100:.1f}%)")
            print(f"     Churned (label=1): {churn_count:,} ({churn_count/len(features)*100:.1f}%)")
        
        # ==============================================================================
        # Add Advanced Features
        # ==============================================================================
        features = self._add_advanced_features(features)
        
        if self.verbose:
            print(f"\n✅ Hoàn thành! Tạo được {features.shape[1]} features cho {features.shape[0]} customers")
        
        return features
    
    def fit_transform(self, transactions_df, y=None):
        """Fit và transform trong một bước"""
        return self.fit(transactions_df, y).transform(transactions_df)
    
    def _calculate_rfm_features(self, transactions_df):
        """Tính toán RFM cơ bản"""
        rfm = transactions_df.groupby('customer_id').agg({
            'transaction_date': ['min', 'max', 'count'],
            'amount': ['sum', 'mean', 'std', 'median']
        })
        rfm.columns = ['first_purchase', 'last_purchase', 'frequency', 
                       'total_spent', 'avg_transaction', 'std_transaction', 'median_transaction']
        rfm = rfm.reset_index()
        
        # Recency: số ngày từ giao dịch cuối đến observation date
        rfm['recency'] = (self.observation_date - rfm['last_purchase']).dt.days
        
        # Monetary: average transaction value
        rfm['monetary'] = rfm['avg_transaction']
        
        # Customer age (tenure): số ngày từ giao dịch đầu đến observation date
        rfm['customer_age_days'] = (self.observation_date - rfm['first_purchase']).dt.days
        
        # Active duration: số ngày từ giao dịch đầu đến giao dịch cuối
        rfm['active_duration'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
        
        # Fill NaN trong std (cho customers chỉ có 1 transaction)
        rfm['std_transaction'] = rfm['std_transaction'].fillna(0)
        
        return rfm
    
    def _add_transaction_stats(self, features, transactions_df):
        """Thêm các thống kê transaction"""
        
        # Purchase frequency rate (transactions per day of customer age)
        features['purchase_frequency_rate'] = features['frequency'] / (features['customer_age_days'] + 1)
        
        # Average days between purchases
        features['avg_days_between_purchases'] = features['active_duration'] / (features['frequency'] - 1 + 1e-6)
        features['avg_days_between_purchases'] = features['avg_days_between_purchases'].clip(lower=0)
        
        # Coefficient of variation (spending volatility)
        features['cv_spending'] = features['std_transaction'] / (features['avg_transaction'] + 1e-6)
        
        # Spending per day (intensity)
        features['spending_per_day'] = features['total_spent'] / (features['customer_age_days'] + 1)
        
        return features
    
    def _add_trend_features(self, features, transactions_df):
        """Thêm features về xu hướng gần đây"""
        
        # Period definitions
        cutoff_recent = self.observation_date - pd.Timedelta(days=30)
        cutoff_previous = cutoff_recent - pd.Timedelta(days=30)
        
        # Recent period (last 30 days before observation)
        recent_tx = transactions_df[
            (transactions_df['transaction_date'] >= cutoff_recent) & 
            (transactions_df['transaction_date'] < self.observation_date)
        ].groupby('customer_id').agg({
            'transaction_date': 'count',
            'amount': ['sum', 'mean']
        })
        recent_tx.columns = ['recent_30d_frequency', 'recent_30d_amount', 'recent_30d_avg']
        
        # Previous period (31-60 days before observation)
        previous_tx = transactions_df[
            (transactions_df['transaction_date'] >= cutoff_previous) & 
            (transactions_df['transaction_date'] < cutoff_recent)
        ].groupby('customer_id').agg({
            'transaction_date': 'count',
            'amount': ['sum', 'mean']
        })
        previous_tx.columns = ['previous_30d_frequency', 'previous_30d_amount', 'previous_30d_avg']
        
        # Last 90 days
        cutoff_90d = self.observation_date - pd.Timedelta(days=90)
        last_90d_tx = transactions_df[
            (transactions_df['transaction_date'] >= cutoff_90d) & 
            (transactions_df['transaction_date'] < self.observation_date)
        ].groupby('customer_id').agg({
            'transaction_date': 'count',
            'amount': 'sum'
        })
        last_90d_tx.columns = ['last_90d_frequency', 'last_90d_amount']
        
        # Merge with features
        features = features.merge(recent_tx, on='customer_id', how='left')
        features = features.merge(previous_tx, on='customer_id', how='left')
        features = features.merge(last_90d_tx, on='customer_id', how='left')
        
        # Fill NaN với 0
        trend_cols = ['recent_30d_frequency', 'recent_30d_amount', 'recent_30d_avg',
                      'previous_30d_frequency', 'previous_30d_amount', 'previous_30d_avg',
                      'last_90d_frequency', 'last_90d_amount']
        features[trend_cols] = features[trend_cols].fillna(0)
        
        # Trend indicators
        features['frequency_trend'] = features['recent_30d_frequency'] - features['previous_30d_frequency']
        features['amount_trend'] = features['recent_30d_amount'] - features['previous_30d_amount']
        
        # Trend ratios (avoid division by zero)
        features['frequency_trend_ratio'] = features['recent_30d_frequency'] / (features['previous_30d_frequency'] + 1)
        features['amount_trend_ratio'] = features['recent_30d_amount'] / (features['previous_30d_amount'] + 1)
        
        # Declining indicators
        features['is_frequency_declining'] = (features['frequency_trend'] < 0).astype(int)
        features['is_amount_declining'] = (features['amount_trend'] < 0).astype(int)
        features['is_declining'] = ((features['frequency_trend'] < 0) | (features['amount_trend'] < 0)).astype(int)
        
        # No activity in recent periods
        features['no_activity_30d'] = (features['recent_30d_frequency'] == 0).astype(int)
        features['no_activity_60d'] = ((features['recent_30d_frequency'] == 0) & 
                                        (features['previous_30d_frequency'] == 0)).astype(int)
        
        return features
    
    def _add_interval_features(self, features, transactions_df):
        """Thêm features về khoảng cách giữa các lần mua"""
        
        interval_stats = []
        
        for customer_id in features['customer_id'].unique():
            customer_tx = transactions_df[
                transactions_df['customer_id'] == customer_id
            ].sort_values('transaction_date')
            
            if len(customer_tx) > 1:
                dates = customer_tx['transaction_date'].values
                intervals = [(pd.to_datetime(dates[i]) - pd.to_datetime(dates[i-1])).days 
                            for i in range(1, len(dates))]
                
                interval_stats.append({
                    'customer_id': customer_id,
                    'mean_interval': np.mean(intervals),
                    'std_interval': np.std(intervals) if len(intervals) > 1 else 0,
                    'min_interval': np.min(intervals),
                    'max_interval': np.max(intervals),
                    'last_interval': intervals[-1] if intervals else 0
                })
            else:
                # Single purchase customers
                interval_stats.append({
                    'customer_id': customer_id,
                    'mean_interval': 0,
                    'std_interval': 0,
                    'min_interval': 0,
                    'max_interval': 0,
                    'last_interval': 0
                })
        
        interval_df = pd.DataFrame(interval_stats)
        features = features.merge(interval_df, on='customer_id', how='left')
        
        # Interval coefficient of variation
        features['cv_interval'] = features['std_interval'] / (features['mean_interval'] + 1e-6)
        
        # Recency compared to mean interval (warning signal)
        features['recency_vs_mean_interval'] = features['recency'] / (features['mean_interval'] + 1e-6)
        
        return features
    
    def _add_rfm_scores(self, features):
        """Tính RFM scores và segments"""
        
        # R score: Lower recency = higher score (5 = best)
        features['R_score'] = pd.qcut(
            features['recency'].rank(method='first'), 
            q=5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        # F score: Higher frequency = higher score
        features['F_score'] = pd.qcut(
            features['frequency'].rank(method='first'), 
            q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # M score: Higher monetary = higher score
        features['M_score'] = pd.qcut(
            features['monetary'].rank(method='first'), 
            q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Combined RFM score
        features['RFM_score'] = features['R_score'] + features['F_score'] + features['M_score']
        
        # High-value flag (RFM >= 12)
        features['is_high_value'] = (features['RFM_score'] >= 12).astype(int)
        
        # Segment assignment
        features['segment'] = features.apply(self._assign_segment, axis=1)
        features['is_at_risk'] = features['segment'].isin(
            ['At Risk', 'About to Sleep', 'Cannot Lose Them', 'Hibernating']
        ).astype(int)
        
        return features
    
    @staticmethod
    def _assign_segment(row):
        """Assign customer segment based on RFM scores"""
        r, f = row['R_score'], row['F_score']
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 4 and f >= 2:
            return 'Loyal'
        elif r >= 4 and f == 1:
            return 'Promising'
        elif r == 3 and f >= 3:
            return 'Potential Loyalist'
        elif r == 3 and f <= 2:
            return 'New Customers'
        elif r == 2 and f >= 3:
            return 'At Risk'
        elif r == 2 and f <= 2:
            return 'About to Sleep'
        elif r == 1 and f >= 3:
            return 'Cannot Lose Them'
        elif r == 1 and f <= 2:
            return 'Hibernating'
        else:
            return 'Need Attention'
    
    def _add_advanced_features(self, features):
        """
        Thêm các feature tương đối (relative features) để model học behavior thay vì số tĩnh
        Các feature này giúp model tổng quát hóa tốt hơn cho các KH khác nhau
        """
        # 1. Recency / Tenure ratio
        # Khách hàng mới mà recency cao thì nguy hiểm hơn khách lâu năm
        features['recency_to_tenure_ratio'] = features['recency'] / (features['customer_age_days'] + 1)
        
        # 2. Recency / Average Purchase Interval
        # Nếu recency > 2-3 lần khoảng cách mua trung bình => dấu hiệu cực nguy hiểm
        # (Đã tính recency_vs_mean_interval trong _add_interval_features)
        
        # 3. Recent activity ratio (30 days vs total)
        features['recent_frequency_ratio'] = features['recent_30d_frequency'] / (features['frequency'] + 1)
        features['recent_amount_ratio'] = features['recent_30d_amount'] / (features['total_spent'] + 1)
        
        # 4. Spending acceleration/deceleration
        # So sánh avg transaction gần đây vs tổng thể
        features['recent_avg_vs_overall'] = features['recent_30d_avg'] / (features['avg_transaction'] + 1)
        
        # 5. Activity consistency (CV of intervals)
        # CV cao = không đều => khó dự đoán, có thể dễ churn
        # (Đã tính cv_interval trong _add_interval_features)
        
        # 6. RFM relative scores
        # Normalized RFM score (0-1)
        features['RFM_score_normalized'] = (features['RFM_score'] - 3) / 12  # Min=3, Max=15 => scale to 0-1
        
        return features
    
    def get_feature_columns(self):
        """Trả về danh sách feature columns cho model training"""
        return self.FEATURE_COLUMNS.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_training_data(transactions_df, cutoff_dates, horizon=60, historical_window=90, verbose=True):
    """
    Tạo tập train bằng cách gộp nhiều snapshot tại các thời điểm khác nhau.
    Sử dụng Sliding Window approach để tránh bias theo thời điểm và tăng số lượng mẫu.
    
    Parameters:
    -----------
    transactions_df : pd.DataFrame
        DataFrame gốc chứa toàn bộ transactions
    cutoff_dates : list
        List các ngày cutoff (observation_date), ví dụ ['2024-01-01', '2024-02-01', '2024-03-01']
    horizon : int
        Cửa sổ dự báo (default: 60 ngày)
        - Dùng để lọc KH active: recency < horizon
        - Dùng để tính label: KH có mua trong horizon ngày tới không?
    historical_window : int
        Số ngày quá khứ để tính features (default: 90)
        Chỉ lấy transactions từ (observation_date - historical_window) đến observation_date
    verbose : bool
        In thông tin chi tiết (default: True)
    
    Returns:
    --------
    pd.DataFrame: Training set được gộp từ nhiều snapshots
    
    Example:
    --------
    >>> # Tạo tập train từ 6 tháng đầu năm
    >>> train_cutoffs = ['2024-01-01', '2024-02-01', '2024-03-01', 
    ...                  '2024-04-01', '2024-05-01', '2024-06-01']
    >>> train_df = generate_training_data(df_transactions, train_cutoffs, horizon=60, historical_window=90)
    >>> 
    >>> # Tạo tập test từ tháng 9
    >>> test_cutoffs = ['2024-09-01']
    >>> test_df = generate_training_data(df_transactions, test_cutoffs)
    """
    all_features = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔄 SLIDING WINDOW - Generating Training Data")
        print(f"{'='*80}")
        print(f"Total cutoff dates: {len(cutoff_dates)}")
        print(f"Horizon: {horizon} days")
        print(f"Historical window: {historical_window} days")
        print(f"{'='*80}\n")
    
    for i, cutoff in enumerate(cutoff_dates, 1):
        if verbose:
            print(f"\n[{i}/{len(cutoff_dates)}] Processing cutoff: {cutoff}")
            print("-" * 60)
        
        # Khởi tạo Pipeline cho từng cutoff
        engineer = ChurnFeatureEngineer(
            observation_date=cutoff,
            horizon=horizon,
            historical_window=historical_window,
            verbose=verbose
        )
        
        # Transform data
        df_snapshot = engineer.transform(transactions_df)
        
        # Thêm cột xác định snapshot để debug/track nếu cần
        df_snapshot['snapshot_date'] = pd.to_datetime(cutoff)
        
        all_features.append(df_snapshot)
    
    # Gộp tất cả snapshots lại thành 1 tập Training lớn
    full_training_set = pd.concat(all_features, ignore_index=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ COMPLETED - Training Data Generated")
        print(f"{'='*80}")
        print(f"Total samples: {len(full_training_set):,}")
        print(f"Total features: {full_training_set.shape[1]}")
        print(f"\nLabel distribution:")
        print(full_training_set['churn_label'].value_counts())
        print(f"{'='*80}\n")
    
    return full_training_set


def segment_by_churn_risk(probability, thresholds=(0.3, 0.5, 0.7)):
    """
    Phân loại khách hàng theo mức độ rủi ro churn
    
    Parameters:
    -----------
    probability : float
        Xác suất churn (0-1)
    thresholds : tuple
        Ngưỡng phân loại (low, medium, high)
    
    Returns:
    --------
    str: Risk segment name
    """
    low, medium, high = thresholds
    if probability < low:
        return 'Low Risk'
    elif probability < medium:
        return 'Medium Risk'
    elif probability < high:
        return 'High Risk'
    else:
        return 'Critical Risk'


def prepare_features_for_model(features_df, feature_columns=None):
    """
    Chuẩn bị features cho model training
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame features từ ChurnFeatureEngineer
    feature_columns : list, optional
        Danh sách columns để sử dụng. Nếu None, sử dụng FEATURE_COLUMNS mặc định
    
    Returns:
    --------
    pd.DataFrame: DataFrame với features đã được xử lý
    """
    if feature_columns is None:
        feature_columns = ChurnFeatureEngineer.FEATURE_COLUMNS
    
    X = features_df[feature_columns].copy()
    
    # Handle NaN và inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X


if __name__ == "__main__":
    # Example usage
    print("ChurnFeatureEngineer - Feature Engineering Pipeline")
    print("=" * 60)
    print(f"Number of features: {len(ChurnFeatureEngineer.FEATURE_COLUMNS)}")
    print("\nFeature groups:")
    print("  - RFM Features")
    print("  - Transaction Statistics")
    print("  - Trend Features")
    print("  - Interval Features")
    print("  - RFM Scores")
    print("\nUsage:")
    print("  from src.feature_engineer import ChurnFeatureEngineer")
    print("  pipeline = ChurnFeatureEngineer(observation_date=pd.Timestamp('2025-10-01'))")
    print("  features = pipeline.fit_transform(transactions_df)")
