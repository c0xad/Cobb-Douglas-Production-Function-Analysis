import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import norm, chi2
from scipy.optimize import minimize
import warnings
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

@dataclass
class VARResults:
    coefficients: Dict[str, np.ndarray]
    residuals: pd.DataFrame
    forecast: pd.DataFrame
    impulse_responses: Dict[str, np.ndarray]
    variance_decomposition: pd.DataFrame
    granger_causality: Dict[str, Dict[str, float]]
    information_criteria: Dict[str, float]

@dataclass
class MarkovSwitchingResults:
    regime_parameters: Dict[str, np.ndarray]
    transition_probs: np.ndarray
    smoothed_probs: pd.DataFrame
    expected_durations: Dict[int, float]
    regime_stats: Dict[int, Dict[str, float]]
    filtered_states: np.ndarray
    aic: float
    bic: float

class DynamicProductionModels:
    def __init__(self, data: pd.DataFrame, frequency: str = 'M'):
        """
        Initialize dynamic production models for time series analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with datetime index
        frequency : str
            Data frequency ('D', 'M', 'Q', 'Y')
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
        
        self.data = data
        self.frequency = frequency
        self.results = {}
        
    def estimate_var(self, variables: List[str], maxlags: int = 12,
                    deterministic: str = 'co', seasons: bool = True) -> VARResults:
        """
        Estimate VAR model with comprehensive diagnostics
        
        Parameters:
        -----------
        variables : List[str]
            List of variable names to include
        maxlags : int
            Maximum number of lags to consider
        deterministic : str
            Deterministic terms ('n','co','lo','li','cili')
        seasons : bool
            Include seasonal dummies
        """
        # Prepare data
        data = self.data[variables].copy()
        
        # Add seasonal dummies if requested
        if seasons:
            if self.frequency == 'M':
                data = pd.concat([data, pd.get_dummies(data.index.month, prefix='month')], axis=1)
            elif self.frequency == 'Q':
                data = pd.concat([data, pd.get_dummies(data.index.quarter, prefix='quarter')], axis=1)
        
        # Test for stationarity
        adf_results = {}
        for var in variables:
            adf_results[var] = adfuller(data[var], regression=deterministic)
        
        # Estimate VAR model
        model = VAR(data[variables])
        
        # Lag order selection
        lag_order = model.select_order(maxlags=maxlags)
        selected_lags = lag_order.selected_orders['aic']
        
        # Fit model
        results = model.fit(maxlags=selected_lags, trend=deterministic)
        
        # Granger causality tests
        granger_matrix = {}
        for v1, v2 in combinations(variables, 2):
            granger_test = results.test_causality(v1, [v2], kind='f')
            granger_matrix[f"{v1}->{v2}"] = {
                'test_statistic': granger_test.test_statistic,
                'p_value': granger_test.pvalue,
                'df': granger_test.df
            }
        
        # Forecast
        forecast_steps = 12 if self.frequency == 'M' else 4
        forecast = results.forecast(data[variables].values[-selected_lags:], forecast_steps)
        forecast_df = pd.DataFrame(forecast, columns=variables)
        
        # Impulse responses
        irf = results.irf(periods=20)
        irf_dict = {
            'orthogonalized': irf.orth_irfs,
            'accumulated': irf.cum_effects,
            'confidence_intervals': irf.irfs_ci()
        }
        
        # Variance decomposition
        fevd = results.fevd(periods=20)
        fevd_df = pd.DataFrame(fevd.decomp, 
                             columns=variables,
                             index=pd.MultiIndex.from_product([variables, range(20)]))
        
        return VARResults(
            coefficients=results.params,
            residuals=pd.DataFrame(results.resid, columns=variables),
            forecast=forecast_df,
            impulse_responses=irf_dict,
            variance_decomposition=fevd_df,
            granger_causality=granger_matrix,
            information_criteria={
                'aic': results.aic,
                'bic': results.bic,
                'hqic': results.hqic,
                'fpe': results.fpe
            }
        )
    
    def estimate_vecm(self, variables: List[str], deterministic: str = 'co',
                     k_ar_diff: int = 2) -> Dict:
        """
        Estimate Vector Error Correction Model
        
        Parameters:
        -----------
        variables : List[str]
            List of variable names
        deterministic : str
            Deterministic terms specification
        k_ar_diff : int
            Number of lagged differences
        """
        data = self.data[variables].copy()
        
        # Test for cointegration
        coint_results = {}
        for v1, v2 in combinations(variables, 2):
            coint_test = coint(data[v1], data[v2])
            coint_results[f"{v1}-{v2}"] = {
                'test_statistic': coint_test[0],
                'p_value': coint_test[1],
                'critical_values': coint_test[2]
            }
        
        # Estimate VECM
        model = VECM(data, k_ar_diff=k_ar_diff, deterministic=deterministic)
        results = model.fit()
        
        # Extract cointegration vectors
        beta = results.beta
        
        # Calculate adjustment speeds
        alpha = results.alpha
        
        # Compute long-run impact matrix
        pi = np.dot(alpha, beta.T)
        
        return {
            'cointegration_tests': coint_results,
            'cointegration_vectors': beta,
            'adjustment_speeds': alpha,
            'long_run_impact': pi,
            'coefficients': results.params,
            'summary': results.summary()
        }
    
    def estimate_markov_switching(self, dependent: str, exog: List[str],
                                k_regimes: int = 2, order: int = 1,
                                switching_variance: bool = True,
                                switching_trend: bool = True) -> MarkovSwitchingResults:
        """
        Estimate Markov Switching Model for regime-dependent production relationships
        
        Parameters:
        -----------
        dependent : str
            Dependent variable name
        exog : List[str]
            List of exogenous variables
        k_regimes : int
            Number of regimes
        order : int
            Autoregressive order
        switching_variance : bool
            Allow for regime-specific variances
        switching_trend : bool
            Allow for regime-specific trends
        """
        # Prepare data
        y = self.data[dependent]
        X = self.data[exog] if exog else None
        
        # Model specification
        model = MarkovRegression(
            y,
            k_regimes=k_regimes,
            order=order,
            trend=switching_trend,
            switching_variance=switching_variance,
            exog=X
        )
        
        # Fit model
        results = model.fit(search_reps=50)
        
        # Extract regime-specific parameters
        regime_params = {}
        for i in range(k_regimes):
            regime_params[f'regime_{i}'] = {
                'mean': results.params[f'regime_{i}'],
                'variance': results.sigma2[i] if switching_variance else results.sigma2
            }
            if X is not None:
                regime_params[f'regime_{i}']['coefficients'] = results.params[f'beta.regime_{i}']
        
        # Calculate regime statistics
        regime_stats = {}
        smoothed_probs = pd.DataFrame(results.smoothed_marginal_probabilities)
        for i in range(k_regimes):
            regime_mask = smoothed_probs[i] > 0.5
            regime_data = y[regime_mask]
            regime_stats[i] = {
                'mean': regime_data.mean(),
                'std': regime_data.std(),
                'duration': results.expected_durations[i],
                'frequency': regime_mask.mean()
            }
        
        return MarkovSwitchingResults(
            regime_parameters=regime_params,
            transition_probs=results.transition_probabilities,
            smoothed_probs=smoothed_probs,
            expected_durations=dict(enumerate(results.expected_durations)),
            regime_stats=regime_stats,
            filtered_states=results.filtered_marginal_probabilities,
            aic=results.aic,
            bic=results.bic
        )
    
    def analyze_productivity_regimes(self, output: str, inputs: List[str],
                                  tfp_method: str = 'solow_residual') -> Dict:
        """
        Analyze productivity regimes using Markov switching models
        
        Parameters:
        -----------
        output : str
            Output variable name
        inputs : List[str]
            Input variable names
        tfp_method : str
            Method for calculating TFP ('solow_residual' or 'index_number')
        """
        # Calculate TFP
        if tfp_method == 'solow_residual':
            # Estimate production function
            X = sm.add_constant(np.log(self.data[inputs]))
            y = np.log(self.data[output])
            model = sm.OLS(y, X).fit()
            
            # Calculate Solow residual
            tfp = y - X @ model.params
            
        else:  # index_number
            # Calculate Törnqvist index
            input_shares = {var: self.data[var] * self.data[f'{var}_price'] for var in inputs}
            total_cost = sum(input_shares.values())
            for var in inputs:
                input_shares[var] = input_shares[var] / total_cost
            
            # Calculate TFP growth
            tfp_growth = np.log(self.data[output]).diff()
            for var in inputs:
                tfp_growth -= 0.5 * (input_shares[var] + input_shares[var].shift(1)) * np.log(self.data[var]).diff()
            
            # Accumulate to levels
            tfp = tfp_growth.cumsum()
        
        # Estimate Markov switching model for TFP
        ms_results = self.estimate_markov_switching(
            dependent=tfp,
            exog=None,
            k_regimes=3,  # Allow for low, medium, and high productivity regimes
            switching_variance=True,
            switching_trend=True
        )
        
        # Analyze regime characteristics
        regime_characteristics = {}
        smoothed_probs = ms_results.smoothed_probs
        
        for regime in range(3):
            regime_mask = smoothed_probs[regime] > 0.5
            regime_data = self.data[regime_mask]
            
            regime_characteristics[regime] = {
                'tfp_mean': tfp[regime_mask].mean(),
                'tfp_std': tfp[regime_mask].std(),
                'output_growth': np.log(regime_data[output]).diff().mean() * 100,
                'input_productivity': {
                    var: (np.log(regime_data[output]) - np.log(regime_data[var])).mean()
                    for var in inputs
                },
                'duration': ms_results.expected_durations[regime],
                'frequency': regime_mask.mean()
            }
        
        return {
            'tfp_series': tfp,
            'markov_switching_results': ms_results,
            'regime_characteristics': regime_characteristics,
            'transition_probabilities': ms_results.transition_probs,
            'regime_timing': smoothed_probs
        }
