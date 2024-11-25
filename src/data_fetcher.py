import wbgapi as wb
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorldBankDataFetcher:
    """
    Fetches and processes economic data from World Bank API
    Indicators used:
    - NY.GDP.MKTP.CN: GDP (current LCU)
    - SL.TLF.TOTL.IN: Total labor force
    - NV.IND.TOTL.CN: Industry (including construction), value added (current LCU)
    - SP.POP.TOTL: Total population
    - FP.CPI.TOTL.ZG: Inflation, consumer prices (annual %)
    """
    
    def __init__(self, country_codes=['JPN'], start_year=2000, end_year=2022):
        self.country_codes = country_codes
        self.start_year = start_year
        self.end_year = end_year
        self.indicators = {
            'gdp': 'NY.GDP.MKTP.CN',
            'labor': 'SL.TLF.TOTL.IN',
            'capital_proxy': 'NV.IND.TOTL.CN',
            'population': 'SP.POP.TOTL',  # Total population
            'inflation': 'FP.CPI.TOTL.ZG'  # Inflation, consumer prices (annual %)
        }
        
    def fetch_data(self):
        logger.info(f"Fetching data for {self.country_codes} from {self.start_year} to {self.end_year}")
        
        data = {country: {} for country in self.country_codes}
        for country in self.country_codes:
            for name, indicator in self.indicators.items():
                try:
                    series = wb.data.DataFrame(
                        indicator,
                        country,
                        time=range(self.start_year, self.end_year + 1),
                        labels=True
                    )
                    # Convert the index to datetime
                    series.index = pd.to_datetime(series.index.astype(str), format='%Y')
                    data[country][name] = series[indicator]
                except Exception as e:
                    logger.error(f"Error fetching {name} for {country}: {str(e)}")
                    raise
        
        # Combine all series into a single DataFrame per country
        combined_data = {}
        for country, indicators_data in data.items():
            df = pd.DataFrame(indicators_data)
            # Handle missing values using interpolation
            df = df.interpolate(method='cubic')
            
            # Calculate derived metrics
            df['Q'] = df['gdp']  # Output (GDP)
            df['L'] = df['labor']  # Labor force
            df['K'] = df['capital_proxy']  # Capital proxy (industrial value added)
            
            # Calculate year-over-year growth rates
            df['Q_growth'] = df['Q'].pct_change() * 100
            df['L_growth'] = df['L'].pct_change() * 100
            df['K_growth'] = df['K'].pct_change() * 100
            
            # Add time trend
            df['time'] = np.arange(len(df))
            
            # Calculate productivity measures
            df['labor_productivity'] = df['Q'] / df['L']
            df['capital_productivity'] = df['Q'] / df['K']
            
            combined_data[country] = df
        
        return combined_data
    
    def preprocess_for_estimation(self, df, apply_smoothing=True, adjust_for_inflation=True):
        """
        Prepare data for Cobb-Douglas estimation with customizable preprocessing options
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with raw economic indicators
        apply_smoothing : bool, default=True
            Whether to apply smoothing to the time series data
        adjust_for_inflation : bool, default=True
            Whether to adjust values for inflation
        
        Returns:
        --------
        pandas.DataFrame
            Processed dataframe ready for estimation
        """
        # Generate preprocessing report
        report = []
        report.append("# Data Preprocessing Report\n")
        
        # Adjust for inflation if requested
        if adjust_for_inflation and 'inflation' in df.columns:
            df['Q_real'] = df['Q'] / (1 + df['inflation']/100)
            df['K_real'] = df['K'] / (1 + df['inflation']/100)
            report.append("## Inflation Adjustment\n")
            report.append(f"- Average inflation rate: {df['inflation'].mean():.2f}%\n")
            df['Q'] = df['Q_real']
            df['K'] = df['K_real']
        
        # Apply smoothing if requested
        if apply_smoothing:
            for col in ['Q', 'K', 'L']:
                original = df[col].copy()
                df[col] = df[col].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                report.append(f"\n## Smoothing Analysis for {col}\n")
                report.append(f"- Original variance: {original.var():.2f}\n")
                report.append(f"- Smoothed variance: {df[col].var():.2f}\n")
        
        # Take natural logarithms
        estimation_df = pd.DataFrame({
            'ln_Q': np.log(df['Q']),
            'ln_L': np.log(df['L']),
            'ln_K': np.log(df['K']),
            'time': df['time']  # For time trend analysis
        })
        
        # Add squared terms for non-linear analysis
        estimation_df['ln_L_squared'] = estimation_df['ln_L'] ** 2
        estimation_df['ln_K_squared'] = estimation_df['ln_K'] ** 2
        estimation_df['ln_LK_interaction'] = estimation_df['ln_L'] * estimation_df['ln_K']
        
        # Calculate growth rates
        for col in ['ln_Q', 'ln_L', 'ln_K']:
            growth = estimation_df[col].diff()
            report.append(f"\n## Growth Analysis for {col}\n")
            report.append(f"- Mean growth rate: {growth.mean():.3f}\n")
            report.append(f"- Growth volatility: {growth.std():.3f}\n")
        
        # Save preprocessing report
        with open('preprocessing_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        return estimation_df
    
    def prepare_and_load_to_analysis(self, country_code=None):
        """
        Fetches, preprocesses, and loads data directly into a CobbDouglasAnalysis instance.
        
        Parameters:
        -----------
        country_code : str, optional
            Specific country code to analyze. If None, uses the first country in country_codes.
            
        Returns:
        --------
        tuple
            (CobbDouglasAnalysis instance, preprocessed DataFrame)
        
        Raises:
        -------
        ValueError
            If data validation fails or preprocessing encounters issues.
        """
        from src.cobb_douglas import CobbDouglasAnalysis
        
        # Determine which country to analyze
        if country_code is None:
            country_code = self.country_codes[0]
        elif country_code not in self.country_codes:
            raise ValueError(f"Country code {country_code} not in initialized country codes: {self.country_codes}")
            
        logger.info(f"Preparing data for Cobb-Douglas analysis for {country_code}")
        
        try:
            # Fetch raw data
            data_dict = self.fetch_data()
            df = data_dict[country_code]
            
            # Additional data validation
            self._validate_data_quality(df)
            
            # Advanced preprocessing steps
            df = self._advanced_preprocessing(df)
            
            # Create and prepare the analysis instance
            analysis = CobbDouglasAnalysis()
            
            # Load data into the analysis instance
            analysis.load_data(
                output=df['Q'].values,
                labor=df['L'].values,
                capital=df['K'].values,
                time=df.index.year.values
            )
            
            logger.info("Successfully prepared and loaded data into CobbDouglasAnalysis instance")
            return analysis, df
            
        except Exception as e:
            logger.error(f"Error in prepare_and_load_to_analysis: {str(e)}")
            raise
            
    def _validate_data_quality(self, df):
        """
        Performs comprehensive data quality checks.
        """
        # Check for negative values
        for col in ['Q', 'L', 'K']:
            if (df[col] <= 0).any():
                raise ValueError(f"Negative or zero values found in {col}. Cobb-Douglas requires strictly positive values.")
        
        # Check for extreme outliers using IQR method
        for col in ['Q', 'L', 'K']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                logger.warning(f"Outliers detected in {col}: {outliers.index.tolist()}")
        
        # Check for unrealistic growth rates
        for col in ['Q_growth', 'L_growth', 'K_growth']:
            extreme_growth = df[abs(df[col]) > 50]  # 50% growth rate threshold
            if not extreme_growth.empty:
                logger.warning(f"Extreme growth rates detected in {col}: {extreme_growth.index.tolist()}")
    
    def _advanced_preprocessing(self, df):
        """
        Applies advanced preprocessing techniques to prepare data for Cobb-Douglas analysis.
        """
        # Calculate moving averages for smoothing
        window_size = 3
        df['Q_MA'] = df['Q'].rolling(window=window_size, center=True).mean()
        df['L_MA'] = df['L'].rolling(window=window_size, center=True).mean()
        df['K_MA'] = df['K'].rolling(window=window_size, center=True).mean()
        
        # Fill NaN values from moving average calculation
        df['Q_MA'] = df['Q_MA'].fillna(df['Q'])
        df['L_MA'] = df['L_MA'].fillna(df['L'])
        df['K_MA'] = df['K_MA'].fillna(df['K'])
        
        # Calculate technical efficiency indicators
        df['capital_intensity'] = df['K'] / df['L']
        df['output_per_worker'] = df['Q'] / df['L']
        df['capital_productivity'] = df['Q'] / df['K']
        
        # Calculate growth acceleration (second derivative)
        df['Q_growth_accel'] = df['Q_growth'].diff()
        df['L_growth_accel'] = df['L_growth'].diff()
        df['K_growth_accel'] = df['K_growth'].diff()
        
        # Adjust for inflation if available
        if 'inflation' in df.columns:
            df['real_Q'] = df['Q'] / (1 + df['inflation']/100)
            df['real_K'] = df['K'] / (1 + df['inflation']/100)
            # Update main variables to use inflation-adjusted values
            df['Q'] = df['real_Q']
            df['K'] = df['real_K']
        
        # Add cyclical component indicators using time-based features
        df['year'] = df.index.year
        df['cycle'] = np.sin(2 * np.pi * df['time'] / len(df))
        
        return df