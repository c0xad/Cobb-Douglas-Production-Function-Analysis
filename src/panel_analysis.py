import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
import statsmodels.api as sm
from statsmodels.regression.linear_model import PanelOLS
from linearmodels.panel import PanelOLS as LinearPanelOLS
from linearmodels.panel import RandomEffects
import logging

logger = logging.getLogger(__name__)

class PanelDataAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize panel data analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data with MultiIndex (entity_id, time)
        """
        self.data = data
        self.model = None
        self.results = None
        self.entity_effects = None
        self.time_effects = None
        
    def estimate_fixed_effects(self, dependent: str, independent: List[str], 
                             entity_effects: bool = True, time_effects: bool = False) -> Dict:
        """
        Estimate fixed effects model
        
        Parameters:
        -----------
        dependent : str
            Name of dependent variable
        independent : List[str]
            List of independent variables
        entity_effects : bool
            Whether to include entity fixed effects
        time_effects : bool
            Whether to include time fixed effects
        """
        # Prepare data
        y = self.data[dependent]
        X = self.data[independent]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Create model
        model = LinearPanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
        self.results = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Store effects
        if entity_effects:
            self.entity_effects = self.results.estimated_effects
        
        return {
            'params': self.results.params,
            'std_errors': self.results.std_errors,
            'rsquared': self.results.rsquared,
            'rsquared_within': self.results.rsquared_within,
            'f_statistic': self.results.f_statistic,
            'entity_effects': self.entity_effects if entity_effects else None
        }
    
    def estimate_random_effects(self, dependent: str, independent: List[str]) -> Dict:
        """
        Estimate random effects model
        
        Parameters:
        -----------
        dependent : str
            Name of dependent variable
        independent : List[str]
            List of independent variables
        """
        # Prepare data
        y = self.data[dependent]
        X = self.data[independent]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Create model
        model = RandomEffects(y, X)
        self.results = model.fit(cov_type='clustered', cluster_entity=True)
        
        return {
            'params': self.results.params,
            'std_errors': self.results.std_errors,
            'rsquared': self.results.rsquared,
            'rsquared_within': self.results.rsquared_within,
            'theta': self.results.theta,
        }
    
    def hausman_test(self, dependent: str, independent: List[str]) -> Dict:
        """
        Perform Hausman test to choose between fixed and random effects
        
        Parameters:
        -----------
        dependent : str
            Name of dependent variable
        independent : List[str]
            List of independent variables
        """
        # Estimate both models
        fe_results = self.estimate_fixed_effects(dependent, independent)
        re_results = self.estimate_random_effects(dependent, independent)
        
        # Calculate test statistic
        b_fe = fe_results['params']
        b_re = re_results['params']
        v_fe = fe_results['std_errors']
        v_re = re_results['std_errors']
        
        # Hausman test statistic
        diff = b_fe - b_re
        var_diff = v_fe - v_re
        h_stat = diff.T @ np.linalg.inv(var_diff) @ diff
        
        # Degrees of freedom
        df = len(independent)
        
        # P-value
        p_value = 1 - stats.chi2.cdf(h_stat, df)
        
        return {
            'statistic': h_stat,
            'p_value': p_value,
            'df': df,
            'recommendation': 'fixed effects' if p_value < 0.05 else 'random effects'
        }
    
    def estimate_time_varying_productivity(self, output: str, inputs: List[str], 
                                         window_size: int = 5) -> pd.DataFrame:
        """
        Estimate time-varying total factor productivity using rolling window regression
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variables
        window_size : int
            Size of rolling window in time periods
        """
        productivity_scores = []
        
        # Group by entity
        for entity, group in self.data.groupby(level=0):
            # Sort by time
            group = group.sort_index(level=1)
            
            # Rolling window regression
            for i in range(len(group) - window_size + 1):
                window = group.iloc[i:i+window_size]
                
                # Prepare data
                y = window[output]
                X = window[inputs]
                X = sm.add_constant(X)
                
                # Estimate model
                model = sm.OLS(y, X)
                results = model.fit()
                
                # Calculate TFP as residual
                tfp = results.resid.mean()
                
                productivity_scores.append({
                    'entity': entity,
                    'time': window.index.get_level_values(1)[-1],
                    'tfp': tfp
                })
        
        return pd.DataFrame(productivity_scores)
    
    def estimate_efficiency_scores(self, output: str, inputs: List[str], 
                                 method: str = 'fixed_effects') -> pd.DataFrame:
        """
        Estimate technical efficiency scores
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            List of input variables
        method : str
            Estimation method ('fixed_effects' or 'random_effects')
        """
        if method == 'fixed_effects':
            results = self.estimate_fixed_effects(output, inputs)
            effects = results['entity_effects']
        else:
            results = self.estimate_random_effects(output, inputs)
            effects = results['params']['constant']
        
        # Transform effects to efficiency scores
        max_effect = effects.max()
        efficiency_scores = np.exp(effects - max_effect)
        
        return pd.DataFrame({
            'entity': efficiency_scores.index,
            'efficiency_score': efficiency_scores.values
        }) 