import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import pymc as pm
import pytensor.tensor as at
from scipy import stats
import arviz as az
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
import xarray as xr

logger = logging.getLogger(__name__)

@dataclass
class HierarchicalResults:
    posterior_samples: Dict[str, np.ndarray]
    model_diagnostics: Dict[str, float]
    random_effects: Dict[str, pd.DataFrame]
    fixed_effects: pd.DataFrame
    model_comparison: Dict[str, float]
    convergence_stats: Dict[str, Dict[str, float]]
    prediction_intervals: Optional[pd.DataFrame] = None

class HierarchicalProductivityModel:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize hierarchical Bayesian model for productivity analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data with entity and time identifiers
        """
        self.data = data
        self.results = {}
        self.trace = None
        self.model = None
        
    def estimate_multilevel_tfp(self, output: str, inputs: List[str],
                              entity_id: str, time_id: str,
                              group_vars: List[str],
                              n_chains: int = 4,
                              tune: int = 2000,
                              draws: int = 5000) -> HierarchicalResults:
        """
        Estimate multilevel TFP model with nested random effects
        
        Parameters:
        -----------
        output : str
            Output variable name
        inputs : List[str]
            Input variable names
        entity_id : str
            Entity identifier
        time_id : str
            Time period identifier
        group_vars : List[str]
            Grouping variables for hierarchical structure
        n_chains : int
            Number of MCMC chains
        tune : int
            Number of tuning iterations
        draws : int
            Number of posterior draws
        """
        # Standardize variables
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(self.data[inputs]),
                        columns=inputs, index=self.data.index)
        y = scaler.fit_transform(self.data[[output]]).flatten()
        
        # Create group indices
        group_indices = {}
        for var in group_vars:
            group_indices[var] = pd.Categorical(self.data[var]).codes
        
        with pm.Model() as hierarchical_model:
            # Hyperpriors for group-level effects
            group_effects = {}
            for var in group_vars:
                n_groups = len(self.data[var].unique())
                
                # Hierarchical structure for each group
                mu = pm.Normal(f'mu_{var}', mu=0, sd=1)
                sigma = pm.HalfCauchy(f'sigma_{var}', beta=1)
                
                # Group-specific random effects
                group_effects[var] = pm.Normal(f'alpha_{var}',
                                             mu=mu,
                                             sd=sigma,
                                             shape=n_groups)
            
            # Input coefficients with hierarchical priors
            beta_mu = pm.Normal('beta_mu', mu=0, sd=1, shape=len(inputs))
            beta_sigma = pm.HalfCauchy('beta_sigma', beta=1, shape=len(inputs))
            beta = pm.Normal('beta', mu=beta_mu, sd=beta_sigma, shape=len(inputs))
            
            # Time-varying components
            time_trend = pm.GaussianRandomWalk('time_trend',
                                             sd=pm.HalfCauchy('trend_sigma', beta=1),
                                             shape=len(self.data[time_id].unique()))
            
            # Construct linear predictor
            linear_pred = 0
            for i, var in enumerate(inputs):
                linear_pred += beta[i] * X[var]
            
            # Add group effects
            for var in group_vars:
                linear_pred += group_effects[var][group_indices[var]]
            
            # Add time trend
            linear_pred += time_trend[pd.Categorical(self.data[time_id]).codes]
            
            # Model error
            sigma_y = pm.HalfCauchy('sigma_y', beta=1)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=linear_pred, sd=sigma_y, observed=y)
            
            # Estimate model
            trace = pm.sample(draws=draws, tune=tune, chains=n_chains, return_inferencedata=True)
        
        # Store results
        self.model = hierarchical_model
        self.trace = trace
        
        # Calculate convergence diagnostics
        convergence = pm.diagnostics.gelman_rubin(trace)
        
        # Extract random effects
        random_effects = {}
        for var in group_vars:
            effects = pd.DataFrame(trace.posterior[f'alpha_{var}'].values.reshape(-1, len(self.data[var].unique())))
            effects.columns = self.data[var].unique()
            random_effects[var] = effects
        
        # Extract fixed effects
        fixed_effects = pd.DataFrame(trace.posterior['beta'].values.reshape(-1, len(inputs)),
                                   columns=inputs)
        
        # Model comparison metrics
        waic = az.waic(trace)
        loo = az.loo(trace)
        
        # Posterior predictive checks
        with hierarchical_model:
            ppc = pm.sample_posterior_predictive(trace)
        
        # Calculate prediction intervals
        pred_intervals = pd.DataFrame({
            'mean': ppc['y_obs'].mean(axis=0),
            'lower': np.percentile(ppc['y_obs'], 2.5, axis=0),
            'upper': np.percentile(ppc['y_obs'], 97.5, axis=0)
        })
        
        return HierarchicalResults(
            posterior_samples={
                'beta': trace.posterior['beta'].values,
                'time_trend': trace.posterior['time_trend'].values,
                **{f'alpha_{var}': trace.posterior[f'alpha_{var}'].values
                   for var in group_vars}
            },
            model_diagnostics={
                'waic': waic.waic,
                'loo': loo.loo,
                'p_waic': waic.p_waic,
                'p_loo': loo.p_loo
            },
            random_effects=random_effects,
            fixed_effects=fixed_effects,
            model_comparison={
                'waic': waic.waic,
                'loo': loo.loo
            },
            convergence_stats={var: {'r_hat': val}
                             for var, val in convergence.items()},
            prediction_intervals=pred_intervals
        )
    
    def analyze_cross_level_interactions(self, base_results: HierarchicalResults,
                                      interaction_vars: List[Tuple[str, str]]) -> Dict:
        """
        Analyze interactions between hierarchical levels
        
        Parameters:
        -----------
        base_results : HierarchicalResults
            Results from hierarchical model estimation
        interaction_vars : List[Tuple[str, str]]
            List of variable pairs to analyze interactions
        """
        interactions = {}
        
        for var1, var2 in interaction_vars:
            if var1 in base_results.random_effects and var2 in base_results.random_effects:
                # Calculate correlation between random effects
                corr = np.corrcoef(base_results.random_effects[var1].mean(),
                                 base_results.random_effects[var2].mean())[0, 1]
                
                # Test for significant interaction
                z_score = np.arctanh(corr) * np.sqrt(len(base_results.random_effects[var1]) - 3)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                interactions[f"{var1}_{var2}"] = {
                    'correlation': corr,
                    'z_score': z_score,
                    'p_value': p_value
                }
        
        return interactions
    
    def decompose_variance(self, results: HierarchicalResults) -> pd.DataFrame:
        """
        Decompose variance components across hierarchical levels
        """
        variance_components = {}
        
        # Total variance
        total_var = results.prediction_intervals['mean'].var()
        
        # Fixed effects variance
        fixed_var = results.fixed_effects.var().sum()
        
        # Random effects variance
        random_var = {}
        for level, effects in results.random_effects.items():
            random_var[level] = effects.var().mean()
        
        # Calculate proportions
        variance_components['fixed_effects'] = fixed_var / total_var
        for level, var in random_var.items():
            variance_components[f'random_effects_{level}'] = var / total_var
        variance_components['residual'] = 1 - sum(variance_components.values())
        
        return pd.DataFrame.from_dict(variance_components, orient='index',
                                    columns=['proportion_variance'])
    
    def predict_random_effects(self, results: HierarchicalResults,
                             new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict random effects for new entities
        """
        predictions = pd.DataFrame(index=new_data.index)
        
        # Extract posterior means
        fixed_effects = results.fixed_effects.mean()
        
        # Predict for each group level
        for level, effects in results.random_effects.items():
            if level in new_data.columns:
                level_effects = effects.mean()
                predictions[f're_{level}'] = new_data[level].map(level_effects)
        
        # Add fixed effects prediction
        for var in fixed_effects.index:
            if var in new_data.columns:
                predictions[f'fe_{var}'] = new_data[var] * fixed_effects[var]
        
        predictions['total_effect'] = predictions.sum(axis=1)
        
        return predictions
