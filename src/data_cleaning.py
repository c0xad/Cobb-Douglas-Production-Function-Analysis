import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedDataCleaner:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize data cleaner
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to clean
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.imputer = None
        self.anomaly_detector = None
        self.cleaning_report = []
        
    def detect_anomalies(self, method: str = 'isolation_forest', 
                        contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect anomalies in the data
        
        Parameters:
        -----------
        method : str
            Detection method ('isolation_forest' or 'robust_covariance')
        contamination : float
            Expected proportion of outliers in the data
        """
        # Prepare numerical data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        X = self.data[numeric_cols]
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies
        if method == 'isolation_forest':
            self.anomaly_detector = IsolationForest(
                contamination=contamination,
                random_state=42
            )
        else:  # robust_covariance
            self.anomaly_detector = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
        
        # Fit and predict
        anomalies = self.anomaly_detector.fit_predict(X_scaled)
        
        # Create anomaly report
        anomaly_report = pd.DataFrame({
            'index': X.index,
            'is_anomaly': anomalies == -1
        })
        
        # Log findings
        n_anomalies = (anomalies == -1).sum()
        self.cleaning_report.append(
            f"Detected {n_anomalies} anomalies using {method} method"
        )
        
        return anomaly_report
    
    def impute_missing_values(self, method: str = 'iterative', 
                            max_iter: int = 10) -> pd.DataFrame:
        """
        Impute missing values using advanced methods
        
        Parameters:
        -----------
        method : str
            Imputation method ('iterative' or 'knn')
        max_iter : int
            Maximum number of iterations for iterative imputation
        """
        # Prepare numerical data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        X = self.data[numeric_cols]
        
        # Choose imputer
        if method == 'iterative':
            self.imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=42,
                estimator=sm.OLS
            )
        else:  # knn
            self.imputer = KNNImputer(n_neighbors=5)
        
        # Fit and transform
        imputed_values = self.imputer.fit_transform(X)
        
        # Create imputed dataframe
        imputed_df = pd.DataFrame(
            imputed_values,
            columns=X.columns,
            index=X.index
        )
        
        # Log imputation statistics
        missing_before = X.isnull().sum().sum()
        missing_after = imputed_df.isnull().sum().sum()
        self.cleaning_report.append(
            f"Imputed {missing_before - missing_after} missing values using {method} method"
        )
        
        return imputed_df
    
    def clean_time_series(self, columns: List[str], 
                         methods: List[str] = ['trend', 'seasonal']) -> pd.DataFrame:
        """
        Clean time series data using decomposition and filtering
        
        Parameters:
        -----------
        columns : List[str]
            Columns to clean
        methods : List[str]
            Cleaning methods to apply
        """
        cleaned_data = self.data.copy()
        
        for col in columns:
            if 'trend' in methods:
                # Remove trend using Hodrick-Prescott filter
                cycle, trend = sm.tsa.filters.hpfilter(self.data[col])
                cleaned_data[f'{col}_detrended'] = cycle
            
            if 'seasonal' in methods:
                # Remove seasonality using seasonal decomposition
                decomposition = sm.tsa.seasonal_decompose(
                    self.data[col],
                    period=12  # Assuming monthly data
                )
                cleaned_data[f'{col}_seasonally_adjusted'] = (
                    self.data[col] - decomposition.seasonal
                )
        
        return cleaned_data
    
    def handle_outliers(self, columns: List[str], 
                       method: str = 'winsorize',
                       limits: Tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
        """
        Handle outliers using various methods
        
        Parameters:
        -----------
        columns : List[str]
            Columns to process
        method : str
            Method to handle outliers ('winsorize' or 'trim')
        limits : Tuple[float, float]
            Lower and upper percentiles for winsorization
        """
        cleaned_data = self.data.copy()
        
        for col in columns:
            if method == 'winsorize':
                # Winsorize the data
                lower = np.percentile(self.data[col], limits[0] * 100)
                upper = np.percentile(self.data[col], limits[1] * 100)
                cleaned_data[col] = self.data[col].clip(lower=lower, upper=upper)
            else:  # trim
                # Trim outliers
                mask = (self.data[col] > np.percentile(self.data[col], limits[0] * 100)) & \
                      (self.data[col] < np.percentile(self.data[col], limits[1] * 100))
                cleaned_data.loc[~mask, col] = np.nan
        
        return cleaned_data
    
    def validate_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality checks
        """
        quality_report = {}
        
        # Check missing values
        missing = self.data.isnull().sum()
        quality_report['missing_values'] = missing[missing > 0].to_dict()
        
        # Check duplicates
        duplicates = self.data.duplicated().sum()
        quality_report['duplicate_rows'] = duplicates
        
        # Check data types
        quality_report['data_types'] = self.data.dtypes.to_dict()
        
        # Check value ranges
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        ranges = {}
        for col in numeric_cols:
            ranges[col] = {
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'mean': self.data[col].mean(),
                'std': self.data[col].std()
            }
        quality_report['value_ranges'] = ranges
        
        return quality_report
    
    def get_cleaning_report(self) -> List[str]:
        """
        Get the cleaning report
        """
        return self.cleaning_report 