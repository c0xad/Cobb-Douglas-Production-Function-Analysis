import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Optional, Union, Tuple
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AdvancedProductionFunctions:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize advanced production function analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing output, labor, and capital
        """
        self.data = data
        self.params = {}
        self.results = None
        self.elasticities = {}
        
    def estimate_leontief(self, output: str, labor: str, capital: str) -> Dict:
        """
        Estimate Leontief production function with fixed proportions
        
        Parameters:
        -----------
        output : str
            Name of output variable
        labor : str
            Name of labor input variable
        capital : str
            Name of capital input variable
        """
        # Calculate input-output ratios
        labor_ratio = self.data[output] / self.data[labor]
        capital_ratio = self.data[output] / self.data[capital]
        
        # Find minimum ratios (technical coefficients)
        a_L = 1 / np.median(labor_ratio)  # Labor coefficient
        a_K = 1 / np.median(capital_ratio)  # Capital coefficient
        
        # Calculate predicted output
        y_pred_labor = self.data[labor] / a_L
        y_pred_capital = self.data[capital] / a_K
        y_pred = np.minimum(y_pred_labor, y_pred_capital)
        
        # Calculate efficiency measures
        total_inefficiency = self.data[output] - y_pred
        labor_slack = y_pred_labor - y_pred
        capital_slack = y_pred_capital - y_pred
        
        # Calculate R-squared
        ss_tot = np.sum((self.data[output] - self.data[output].mean()) ** 2)
        ss_res = np.sum((self.data[output] - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        self.params = {
            'labor_coefficient': a_L,
            'capital_coefficient': a_K,
            'r_squared': r_squared
        }
        
        return {
            'parameters': self.params,
            'predictions': y_pred,
            'inefficiencies': {
                'total': total_inefficiency,
                'labor_slack': labor_slack,
                'capital_slack': capital_slack
            }
        }
    
    def estimate_dynamic_ces(self, output: str, labor: str, capital: str,
                           time_var: str, sectors: Optional[List[str]] = None) -> Dict:
        """
        Estimate CES production function with time-varying elasticity of substitution
        
        Parameters:
        -----------
        output : str
            Name of output variable
        labor : str
            Name of labor input variable
        capital : str
            Name of capital input variable
        time_var : str
            Name of time variable
        sectors : Optional[List[str]]
            List of sector identifiers for sector-specific estimation
        """
        def ces_objective(params, y, l, k, t):
            A, delta, rho_base, rho_trend = params
            # Time-varying elasticity of substitution
            rho_t = rho_base + rho_trend * t
            sigma_t = 1 / (1 + rho_t)  # Elasticity of substitution
            
            # CES production function
            y_pred = A * (delta * l ** (-rho_t) + (1 - delta) * k ** (-rho_t)) ** (-1/rho_t)
            return np.sum((y - y_pred) ** 2)
        
        results = {}
        if sectors is not None:
            # Sector-specific estimation
            for sector in sectors:
                sector_mask = self.data['sector'] == sector
                sector_data = self.data[sector_mask]
                
                # Initial parameter guess
                x0 = [1.0, 0.5, 0.5, 0.01]
                
                # Optimize
                res = minimize(
                    ces_objective,
                    x0,
                    args=(
                        sector_data[output].values,
                        sector_data[labor].values,
                        sector_data[capital].values,
                        sector_data[time_var].values
                    ),
                    method='Nelder-Mead'
                )
                
                # Calculate time-varying elasticities
                rho_t = res.x[2] + res.x[3] * sector_data[time_var].values
                sigma_t = 1 / (1 + rho_t)
                
                results[sector] = {
                    'A': res.x[0],
                    'delta': res.x[1],
                    'rho_base': res.x[2],
                    'rho_trend': res.x[3],
                    'elasticity_mean': np.mean(sigma_t),
                    'elasticity_std': np.std(sigma_t),
                    'convergence': res.success
                }
        else:
            # Pooled estimation
            x0 = [1.0, 0.5, 0.5, 0.01]
            res = minimize(
                ces_objective,
                x0,
                args=(
                    self.data[output].values,
                    self.data[labor].values,
                    self.data[capital].values,
                    self.data[time_var].values
                ),
                method='Nelder-Mead'
            )
            
            # Calculate time-varying elasticities
            rho_t = res.x[2] + res.x[3] * self.data[time_var].values
            sigma_t = 1 / (1 + rho_t)
            
            results['pooled'] = {
                'A': res.x[0],
                'delta': res.x[1],
                'rho_base': res.x[2],
                'rho_trend': res.x[3],
                'elasticity_mean': np.mean(sigma_t),
                'elasticity_std': np.std(sigma_t),
                'convergence': res.success
            }
        
        self.params = results
        return results
    
    def calculate_marginal_products(self, output: str, labor: str, capital: str,
                                  model_type: str = 'leontief', time_var: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Calculate marginal products for different production function specifications
        
        Parameters:
        -----------
        output : str
            Name of output variable
        labor : str
            Name of labor input variable
        capital : str
            Name of capital input variable
        model_type : str
            Type of production function ('leontief' or 'ces')
        time_var : Optional[str]
            Name of time variable, required for CES model
        """
        if model_type == 'leontief':
            # Marginal products are either 0 or 1/coefficient depending on which input is binding
            labor_binding = (self.data[labor] / self.params['labor_coefficient'] <=
                           self.data[capital] / self.params['capital_coefficient'])
            
            mp_labor = np.where(labor_binding, 1/self.params['labor_coefficient'], 0)
            mp_capital = np.where(~labor_binding, 1/self.params['capital_coefficient'], 0)
            
        else:  # ces
            # For CES, calculate numerical derivatives
            epsilon = 1e-6
            for sector, params in self.params.items():
                rho_t = params['rho_base'] + params['rho_trend'] * self.data[time_var].values
                
                # Labor marginal product
                l_plus = self.data[labor] + epsilon
                y_plus = params['A'] * (params['delta'] * l_plus ** (-rho_t) +
                                      (1 - params['delta']) * self.data[capital] ** (-rho_t)) ** (-1/rho_t)
                mp_labor = (y_plus - self.data[output]) / epsilon
                
                # Capital marginal product
                k_plus = self.data[capital] + epsilon
                y_plus = params['A'] * (params['delta'] * self.data[labor] ** (-rho_t) +
                                      (1 - params['delta']) * k_plus ** (-rho_t)) ** (-1/rho_t)
                mp_capital = (y_plus - self.data[output]) / epsilon
        
        return {
            'MP_L': mp_labor,
            'MP_K': mp_capital
        }
    
    def test_returns_to_scale(self, output: str, labor: str, capital: str,
                            model_type: str = 'leontief', time_var: Optional[str] = None) -> Dict:
        """
        Test for returns to scale in different production function specifications
        """
        # Calculate elasticity of scale
        lambda_factor = 1.1  # 10% increase in inputs
        
        if model_type == 'leontief':
            # For Leontief, scale elasticity is 1 by construction
            scale_elasticity = 1.0
            regime = 'constant'
            
        else:  # ces
            if time_var is None:
                raise ValueError("time_var is required for CES model")
            
            # Calculate output with scaled inputs
            scale_elasticities = []
            for sector, params in self.params.items():
                rho_t = params['rho_base'] + params['rho_trend'] * self.data[time_var].values
                
                y_base = self.data[output].values
                y_scaled = (params['A'] * 
                           (params['delta'] * (lambda_factor * self.data[labor].values) ** (-rho_t) +
                            (1 - params['delta']) * (lambda_factor * self.data[capital].values) ** (-rho_t)) ** (-1/rho_t))
                
                sector_elasticity = np.mean(np.log(y_scaled/y_base) / np.log(lambda_factor))
                scale_elasticities.append(sector_elasticity)
            
            # Take mean across sectors
            scale_elasticity = np.mean(scale_elasticities)
            
            # Determine returns to scale regime
            if np.abs(scale_elasticity - 1) < 0.05:
                regime = 'constant'
            elif scale_elasticity > 1:
                regime = 'increasing'
            else:
                regime = 'decreasing'
        
        return {
            'scale_elasticity': float(scale_elasticity),
            'regime': regime,
            'lambda_factor': lambda_factor
        } 