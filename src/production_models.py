import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statsmodels.api as sm
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

@dataclass
class TranslogResults:
    coefficients: Dict[str, float]
    elasticities: Dict[str, np.ndarray]
    returns_to_scale: np.ndarray
    r_squared: float
    std_errors: Dict[str, float]

class AdvancedProductionModels:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize advanced production models including Translog and Hybrid specifications
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing output, inputs, and optional time variables
        """
        self.data = data
        self.results = {}
        
    def estimate_translog(self, output: str, inputs: List[str], time_var: Optional[str] = None) -> TranslogResults:
        """
        Estimate Translog production function with optional time-varying parameters
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variable names
        time_var : Optional[str]
            Name of time variable for dynamic specification
            
        Returns:
        --------
        TranslogResults containing estimated parameters and diagnostics
        """
        # Take logarithms of variables
        log_y = np.log(self.data[output])
        log_inputs = {var: np.log(self.data[var]) for var in inputs}
        
        # Create interaction terms and squared terms
        X_terms = []
        X_names = []
        
        # First-order terms
        for var in inputs:
            X_terms.append(log_inputs[var])
            X_names.append(f'ln_{var}')
        
        # Second-order terms
        for i, var1 in enumerate(inputs):
            for var2 in inputs[i:]:
                X_terms.append(log_inputs[var1] * log_inputs[var2])
                X_names.append(f'ln_{var1}_ln_{var2}')
        
        # Time interactions if specified
        if time_var:
            t = self.data[time_var]
            X_terms.extend([t, t**2])
            X_names.extend(['t', 't2'])
            
            for var in inputs:
                X_terms.append(t * log_inputs[var])
                X_names.append(f't_ln_{var}')
        
        # Combine terms and estimate
        X = np.column_stack(X_terms)
        X = sm.add_constant(X)
        X_names.insert(0, 'const')
        
        # Estimate model
        model = sm.OLS(log_y, X)
        results = model.fit()
        
        # Calculate elasticities
        elasticities = {}
        for i, var in enumerate(inputs):
            # Base elasticity
            elast = results.params[f'ln_{var}']
            
            # Add interaction terms
            for var2 in inputs:
                if f'ln_{var}_ln_{var2}' in X_names:
                    elast += results.params[f'ln_{var}_ln_{var2}'] * log_inputs[var2]
                elif f'ln_{var2}_ln_{var}' in X_names:
                    elast += results.params[f'ln_{var2}_ln_{var}'] * log_inputs[var2]
            
            # Add time interaction if present
            if time_var and f't_ln_{var}' in X_names:
                elast += results.params[f't_ln_{var}'] * self.data[time_var]
            
            elasticities[var] = elast
        
        # Calculate returns to scale
        rts = sum(elasticities.values())
        
        return TranslogResults(
            coefficients=dict(zip(X_names, results.params)),
            elasticities=elasticities,
            returns_to_scale=rts,
            r_squared=results.rsquared,
            std_errors=dict(zip(X_names, results.bse))
        )
    
    def estimate_hybrid_model(self, output: str, inputs: List[str], 
                            specification: str = 'cd_ces') -> Dict:
        """
        Estimate hybrid production function combining elements of different specifications
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variable names
        specification : str
            Type of hybrid model ('cd_ces' for Cobb-Douglas-CES hybrid)
            
        Returns:
        --------
        Dictionary containing estimated parameters and model statistics
        """
        if specification == 'cd_ces':
            def hybrid_objective(params):
                A, alpha, beta, rho = params
                
                # CES component for capital inputs
                if abs(rho) < 1e-6:  # Close to Cobb-Douglas case
                    capital_component = np.prod([self.data[k]**beta[i] 
                                              for i, k in enumerate(inputs[1:])])
                else:
                    capital_component = (sum(beta[i] * self.data[k]**(-rho)
                                          for i, k in enumerate(inputs[1:])))**(-1/rho)
                
                # Cobb-Douglas component for labor
                labor_component = self.data[inputs[0]]**alpha
                
                # Combined production function
                y_pred = A * labor_component * capital_component
                
                # Loss function (negative log-likelihood)
                return np.sum((np.log(self.data[output]) - np.log(y_pred))**2)
            
            # Initial parameter guesses
            n_inputs = len(inputs)
            x0 = np.concatenate([[1.0, 0.3], np.repeat(0.3, n_inputs-1), [0.5]])
            
            # Parameter bounds
            bounds = [(0.01, None), (0.01, 0.99)] + [(0.01, 0.99)]*(n_inputs-1) + [(-0.99, 0.99)]
            
            # Optimize
            result = minimize(hybrid_objective, x0, bounds=bounds, method='L-BFGS-B')
            
            # Extract parameters
            A, alpha = result.x[:2]
            beta = result.x[2:-1]
            rho = result.x[-1]
            
            # Calculate elasticity of substitution
            sigma = 1 / (1 + rho)
            
            return {
                'technology': A,
                'labor_elasticity': alpha,
                'capital_elasticities': dict(zip(inputs[1:], beta)),
                'substitution_parameter': rho,
                'elasticity_substitution': sigma,
                'convergence': result.success,
                'objective_value': result.fun
            }
        else:
            raise ValueError(f"Unsupported hybrid specification: {specification}")
    
    def estimate_ho_model(self, output: str, inputs: List[str], 
                         factor_intensities: Dict[str, float]) -> Dict:
        """
        Estimate Heckscher-Ohlin inspired production model
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variable names
        factor_intensities : Dict[str, float]
            Dictionary of factor intensity parameters for each input
            
        Returns:
        --------
        Dictionary containing estimated parameters and trade-related metrics
        """
        # Take logarithms
        log_y = np.log(self.data[output])
        log_X = np.column_stack([np.log(self.data[x]) for x in inputs])
        
        # Add factor intensity interactions
        intensity_terms = []
        for var, intensity in factor_intensities.items():
            intensity_terms.append(np.log(self.data[var]) * intensity)
        
        X = np.column_stack([log_X] + intensity_terms)
        X = sm.add_constant(X)
        
        # Estimate model
        model = sm.OLS(log_y, X)
        results = model.fit()
        
        # Calculate factor price equalization implications
        factor_prices = {}
        for i, var in enumerate(inputs):
            factor_prices[var] = results.params[i+1] * self.data[output] / self.data[var]
        
        # Calculate Stolper-Samuelson elasticities
        ss_elasticities = {}
        for i, var in enumerate(inputs):
            ss_elasticities[var] = results.params[i+1] * (1 + factor_intensities.get(var, 0))
        
        return {
            'coefficients': dict(zip(['const'] + inputs + 
                                   [f'{k}_intensity' for k in factor_intensities.keys()],
                                   results.params)),
            'factor_prices': factor_prices,
            'stolper_samuelson_elasticities': ss_elasticities,
            'r_squared': results.rsquared,
            'std_errors': dict(zip(['const'] + inputs + 
                                 [f'{k}_intensity' for k in factor_intensities.keys()],
                                 results.bse))
        }
