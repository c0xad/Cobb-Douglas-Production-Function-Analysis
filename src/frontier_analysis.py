import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm, jarque_bera, kstest
from typing import Dict, List, Optional, Union, Tuple
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
import warnings
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

logger = logging.getLogger(__name__)

class StochasticFrontierAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize stochastic frontier analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing output and inputs
        """
        self.data = data
        self.params = {}
        self.efficiency_scores = None
        self.frontier_residuals = None
        self.confidence_intervals = None
        self.model_diagnostics = {}
        self._likelihood_values = {}
        
    def estimate_production_frontier(self, output: str, inputs: List[str],
                                   time_effects: bool = True,
                                   distribution: str = 'half_normal') -> Dict:
        """
        Estimate stochastic production frontier
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variables
        time_effects : bool
            Whether to include time effects
        distribution : str
            Inefficiency distribution ('half_normal', 'exponential', or 'truncated_normal')
            
        Returns:
        --------
        Dict containing estimation results, diagnostics, and model fit statistics
        """
        # Prepare data
        y = np.log(self.data[output])
        X = sm.add_constant(np.log(self.data[inputs]))
        
        if time_effects:
            X = pd.concat([X, pd.get_dummies(self.data.index.year, prefix='year')], axis=1)
        
        def likelihood_function(params):
            beta = params[:X.shape[1]]
            sigma_v = np.exp(params[-2])  # noise std
            sigma_u = np.exp(params[-1])  # inefficiency std
            
            # Production frontier
            epsilon = y - X @ beta
            
            if distribution == 'half_normal':
                # Log-likelihood for half-normal inefficiency
                lambda_param = sigma_u / sigma_v
                sigma = np.sqrt(sigma_u**2 + sigma_v**2)
                
                ll = (-0.5 * np.log(2*np.pi) - np.log(sigma) +
                     norm.logcdf(-epsilon*lambda_param/sigma) -
                     0.5*(epsilon/sigma)**2)
                
            elif distribution == 'exponential':
                # Log-likelihood for exponential inefficiency
                lambda_param = 1 / sigma_u
                
                ll = np.log(lambda_param) - epsilon*lambda_param - 0.5*(epsilon/sigma_v)**2
                
            else:  # truncated_normal
                mu = params[-3]  # mean of pre-truncated distribution
                
                # Log-likelihood for truncated normal inefficiency
                lambda_param = sigma_u / sigma_v
                sigma = np.sqrt(sigma_u**2 + sigma_v**2)
                
                ll = (-0.5 * np.log(2*np.pi) - np.log(sigma) +
                     truncnorm.logpdf(epsilon, a=0, b=np.inf, loc=mu, scale=sigma))
            
            return -np.sum(ll)
        
        # Initial parameter values
        n_params = X.shape[1] + (3 if distribution == 'truncated_normal' else 2)
        x0 = np.zeros(n_params)
        x0[-2:] = -1  # log of initial standard deviations
        
        if distribution == 'truncated_normal':
            x0[-3] = 0  # initial mean
        
        # Optimize likelihood function
        result = minimize(likelihood_function, x0, method='Nelder-Mead')
        
        # Store parameters
        self.params = {
            'beta': result.x[:X.shape[1]],
            'sigma_v': np.exp(result.x[-2]),
            'sigma_u': np.exp(result.x[-1])
        }
        
        if distribution == 'truncated_normal':
            self.params['mu'] = result.x[-3]
        
        # Calculate residuals
        self.frontier_residuals = y - X @ self.params['beta']
        
        # Calculate technical efficiency scores with confidence intervals
        self.efficiency_scores, self.confidence_intervals = self._calculate_efficiency_scores_with_ci(distribution)
        
        # Compute model diagnostics
        self._compute_model_diagnostics(X, y, distribution, result)
        
        return {
            'parameters': self.params,
            'efficiency_scores': self.efficiency_scores,
            'confidence_intervals': self.confidence_intervals,
            'convergence': result.success,
            'log_likelihood': -result.fun,
            'diagnostics': self.model_diagnostics
        }
    
    def _compute_model_diagnostics(self, X: pd.DataFrame, y: pd.Series, 
                                 distribution: str, optimization_result) -> None:
        """
        Compute comprehensive model diagnostics
        """
        n_params = len(optimization_result.x)
        n_obs = len(y)
        
        # Information criteria
        log_likelihood = -optimization_result.fun
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood
        
        # Residual diagnostics
        residuals = self.frontier_residuals
        
        # Jarque-Bera test for normality
        jb_stat, jb_pval = jarque_bera(residuals)
        
        # Breusch-Pagan test for heteroscedasticity
        bp_stat, bp_pval = het_breuschpagan(residuals, X)
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(residuals)
        
        # Gamma parameter (variance decomposition)
        gamma = self.params['sigma_u']**2 / (self.params['sigma_u']**2 + self.params['sigma_v']**2)
        
        # Store diagnostics
        self.model_diagnostics = {
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_pval': bp_pval,
            'durbin_watson': dw_stat,
            'gamma': gamma,
            'n_parameters': n_params,
            'n_observations': n_obs
        }
        
    def _calculate_efficiency_scores_with_ci(self, distribution: str, 
                                           alpha: float = 0.05) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate technical efficiency scores with confidence intervals
        
        Parameters:
        -----------
        distribution : str
            Inefficiency distribution used in estimation
        alpha : float
            Significance level for confidence intervals
        
        Returns:
        --------
        Tuple containing point estimates and confidence intervals
        """
        sigma_v = self.params['sigma_v']
        sigma_u = self.params['sigma_u']
        epsilon = self.frontier_residuals
        
        if distribution == 'half_normal':
            sigma = np.sqrt(sigma_u**2 + sigma_v**2)
            lambda_param = sigma_u / sigma_v
            phi_star = norm.pdf(-epsilon*lambda_param/sigma)
            Phi_star = norm.cdf(-epsilon*lambda_param/sigma)
            
            E_u = sigma_u * sigma_v / sigma * (phi_star / Phi_star)
            var_u = (sigma_u**2 * sigma_v**2 / sigma**2) * (1 + lambda_param * phi_star / Phi_star - (phi_star / Phi_star)**2)
            
        elif distribution == 'exponential':
            E_u = sigma_u * (1 - norm.pdf(epsilon/sigma_v) / (1 - norm.cdf(epsilon/sigma_v)))
            var_u = sigma_u**2
            
        else:  # truncated_normal
            mu = self.params['mu']
            sigma = np.sqrt(sigma_u**2 + sigma_v**2)
            mu_star = (-sigma_u**2 * epsilon + sigma_v**2 * mu) / sigma**2
            sigma_star = sigma_u * sigma_v / sigma
            
            phi_star = norm.pdf((mu_star/sigma_star))
            Phi_star = norm.cdf((mu_star/sigma_star))
            
            E_u = mu_star + sigma_star * (phi_star / Phi_star)
            var_u = sigma_star**2 * (1 + (mu_star/sigma_star) * phi_star/Phi_star - (phi_star/Phi_star)**2)
        
        # Calculate confidence intervals
        z_value = norm.ppf(1 - alpha/2)
        lower_ci = np.exp(-(E_u + z_value * np.sqrt(var_u)))
        upper_ci = np.exp(-(E_u - z_value * np.sqrt(var_u)))
        point_estimates = np.exp(-E_u)
        
        ci_df = pd.DataFrame({
            'lower': lower_ci,
            'upper': upper_ci
        }, index=self.data.index)
        
        return point_estimates, ci_df
    
    def plot_efficiency_scores(self, output_path: Optional[str] = None,
                             plot_type: str = 'time_series') -> None:
        """
        Plot efficiency scores with confidence intervals
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the plot
        plot_type : str
            Type of plot ('time_series', 'histogram', or 'boxplot')
        """
        if self.efficiency_scores is None:
            raise ValueError("Must estimate frontier before plotting efficiency scores")
            
        plt.figure(figsize=(12, 6))
        
        if plot_type == 'time_series':
            plt.plot(self.data.index, self.efficiency_scores, 'b-', label='Efficiency Score')
            plt.fill_between(self.data.index,
                           self.confidence_intervals['lower'],
                           self.confidence_intervals['upper'],
                           alpha=0.2, color='b', label='95% CI')
            plt.xlabel('Time')
            plt.ylabel('Technical Efficiency')
            plt.title('Technical Efficiency Scores Over Time')
            
        elif plot_type == 'histogram':
            sns.histplot(data=self.efficiency_scores, bins=30, kde=True)
            plt.xlabel('Technical Efficiency')
            plt.ylabel('Frequency')
            plt.title('Distribution of Technical Efficiency Scores')
            
        elif plot_type == 'boxplot':
            data_by_year = pd.DataFrame({
                'Year': self.data.index.year,
                'Efficiency': self.efficiency_scores
            })
            sns.boxplot(x='Year', y='Efficiency', data=data_by_year)
            plt.xlabel('Year')
            plt.ylabel('Technical Efficiency')
            plt.title('Technical Efficiency Scores by Year')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_diagnostics(self, output_path: Optional[str] = None) -> None:
        """
        Plot model diagnostics including residual analysis
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the plot
        """
        if self.frontier_residuals is None:
            raise ValueError("Must estimate frontier before plotting diagnostics")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # QQ plot
        sm.graphics.qqplot(self.frontier_residuals, line='45', ax=axes[0,0])
        axes[0,0].set_title('Normal Q-Q Plot')
        
        # Residual histogram
        sns.histplot(self.frontier_residuals, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Residual Distribution')
        
        # Residuals vs Fitted
        fitted_values = self.data[self.data.columns[0]] - self.frontier_residuals
        axes[1,0].scatter(fitted_values, self.frontier_residuals, alpha=0.5)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Fitted Values')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residuals vs Fitted')
        
        # Time series plot of residuals
        axes[1,1].plot(self.data.index, self.frontier_residuals)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Residuals Over Time')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def decompose_tfp_growth(self, output: str, inputs: List[str]) -> pd.DataFrame:
        """
        Decompose total factor productivity growth into technical change,
        efficiency change, and scale efficiency change
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variables
        """
        if self.efficiency_scores is None:
            raise ValueError("Must estimate frontier before decomposing TFP growth")
        
        # Calculate year-over-year changes
        efficiency_change = pd.Series(self.efficiency_scores).pct_change()
        
        # Technical change (shift in frontier)
        if 'year' in self.data.columns:
            year_effects = pd.DataFrame({
                col: self.params['beta'][i]
                for i, col in enumerate(self.data.columns)
                if col.startswith('year')
            })
            technical_change = year_effects.diff().mean(axis=1)
        else:
            technical_change = pd.Series(0, index=self.data.index)
        
        # Scale efficiency change
        output_elasticities = self.params['beta'][1:len(inputs)+1]
        returns_to_scale = np.sum(output_elasticities)
        
        input_growth = np.zeros(len(self.data))
        for i, input_var in enumerate(inputs):
            input_growth += output_elasticities[i] * self.data[input_var].pct_change()
        
        scale_efficiency_change = (returns_to_scale - 1) * input_growth
        
        # Total factor productivity growth
        tfp_growth = (technical_change + efficiency_change + scale_efficiency_change)
        
        return pd.DataFrame({
            'tfp_growth': tfp_growth,
            'technical_change': technical_change,
            'efficiency_change': efficiency_change,
            'scale_efficiency_change': scale_efficiency_change
        })
    
    def test_returns_to_scale(self, confidence_level: float = 0.95) -> Dict:
        """
        Test for constant returns to scale in the production frontier
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for hypothesis test
        """
        # Extract coefficients for inputs
        input_coefficients = self.params['beta'][1:]  # exclude constant
        sum_coefficients = np.sum(input_coefficients)
        
        # Calculate standard error of sum (assuming independence)
        se_sum = np.sqrt(np.sum(self.params.get('std_errors', np.ones_like(input_coefficients))[1:]**2))
        
        # Calculate test statistic
        t_stat = (sum_coefficients - 1) / se_sum
        
        # Critical value
        critical_value = norm.ppf(1 - (1 - confidence_level)/2)
        
        return {
            'sum_coefficients': sum_coefficients,
            'standard_error': se_sum,
            't_statistic': t_stat,
            'critical_value': critical_value,
            'reject_crs': abs(t_stat) > critical_value,
            'p_value': 2 * (1 - norm.cdf(abs(t_stat)))
        }