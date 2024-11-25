import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import networkx as nx
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS
import logging
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TFPDecompositionResults:
    scale_effects: pd.DataFrame
    technical_diffusion: Dict[str, pd.DataFrame]
    input_efficiency: Dict[str, pd.DataFrame]
    spillover_network: nx.DiGraph
    variance_decomposition: pd.DataFrame
    diagnostics: Dict[str, float]

class AdvancedTFPDecomposition:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize advanced TFP decomposition analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data with entity and time identifiers
        """
        self.data = data
        self.results = {}
        
    def decompose_tfp_components(self, output: str, inputs: List[str],
                               entity_id: str, time_id: str,
                               spatial_weights: Optional[pd.DataFrame] = None,
                               bandwidth: float = 0.5) -> TFPDecompositionResults:
        """
        Decompose TFP into scale effects, technical diffusion, and input-specific efficiency
        
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
        spatial_weights : Optional[pd.DataFrame]
            Spatial weights matrix for spillover analysis
        bandwidth : float
            Bandwidth parameter for kernel estimation
        """
        # Calculate base TFP using Solow residual
        tfp = self._calculate_base_tfp(output, inputs)
        
        # Analyze scale effects
        scale_effects = self._analyze_scale_effects(output, inputs, tfp)
        
        # Estimate technical diffusion
        diffusion = self._estimate_technical_diffusion(tfp, entity_id, time_id,
                                                     spatial_weights, bandwidth)
        
        # Calculate input-specific efficiency
        input_efficiency = self._calculate_input_efficiency(output, inputs,
                                                          entity_id, time_id)
        
        # Construct spillover network
        spillover_network = self._construct_spillover_network(diffusion['spillovers'])
        
        # Decompose variance
        variance_decomp = self._decompose_variance_components(
            scale_effects, diffusion, input_efficiency
        )
        
        return TFPDecompositionResults(
            scale_effects=scale_effects,
            technical_diffusion=diffusion,
            input_efficiency=input_efficiency,
            spillover_network=spillover_network,
            variance_decomposition=variance_decomp,
            diagnostics=self._calculate_diagnostics()
        )
    
    def _calculate_base_tfp(self, output: str, inputs: List[str]) -> pd.Series:
        """Calculate base TFP using Solow residual method"""
        # Take logarithms
        log_y = np.log(self.data[output])
        log_X = np.log(self.data[inputs])
        
        # Estimate production function
        model = sm.OLS(log_y, sm.add_constant(log_X))
        results = model.fit()
        
        # Calculate TFP as residual
        tfp = log_y - sm.add_constant(log_X) @ results.params
        
        return pd.Series(tfp, index=self.data.index)
    
    def _analyze_scale_effects(self, output: str, inputs: List[str],
                             tfp: pd.Series) -> pd.DataFrame:
        """Analyze scale effects in productivity"""
        # Calculate input aggregates
        input_agg = np.log(self.data[inputs].sum(axis=1))
        
        # Non-parametric scale effects
        def local_linear_regression(x, y, bandwidth):
            weights = np.exp(-(x[:, None] - x[None, :])**2 / (2 * bandwidth**2))
            return np.array([
                sm.WLS(y, sm.add_constant(x - x0),
                      weights=weights[:, i]).fit().predict([1, 0])
                for i, x0 in enumerate(x)
            ])
        
        # Estimate scale elasticity
        scale_elasticity = local_linear_regression(input_agg.values,
                                                 tfp.values, bandwidth=0.5)
        
        # Test for increasing returns
        def test_increasing_returns(y, X, bandwidth):
            n = len(y)
            h = bandwidth * np.std(X)
            kernel = lambda u: np.exp(-u**2/2) / np.sqrt(2*np.pi)
            
            # Local constant estimator
            m_hat = np.zeros(n)
            for i in range(n):
                weights = kernel((X - X[i])/h)
                m_hat[i] = np.sum(weights * y) / np.sum(weights)
            
            # Test statistic
            T_n = np.sqrt(n*h) * np.mean(np.diff(m_hat))
            p_value = 1 - norm.cdf(T_n)
            
            return {'test_statistic': T_n, 'p_value': p_value}
        
        irs_test = test_increasing_returns(tfp.values, input_agg.values, 0.5)
        
        return pd.DataFrame({
            'scale_elasticity': scale_elasticity,
            'input_aggregate': input_agg,
            'tfp': tfp,
            'irs_test_stat': irs_test['test_statistic'],
            'irs_p_value': irs_test['p_value']
        })
    
    def _estimate_technical_diffusion(self, tfp: pd.Series, entity_id: str,
                                    time_id: str, spatial_weights: Optional[pd.DataFrame],
                                    bandwidth: float) -> Dict[str, pd.DataFrame]:
        """Estimate technical diffusion and spillover effects"""
        # Prepare spatial weights if not provided
        if spatial_weights is None:
            entities = self.data[entity_id].unique()
            n = len(entities)
            spatial_weights = pd.DataFrame(1/n, index=entities, columns=entities)
            np.fill_diagonal(spatial_weights.values, 0)
        
        # Calculate spatial lags
        tfp_matrix = tfp.unstack(entity_id)
        spatial_lag = spatial_weights @ tfp_matrix
        
        # Estimate diffusion parameters
        def estimate_diffusion_params(y, X, W):
            n, t = X.shape
            
            # Spatial Durbin model
            X_aug = np.vstack([X, W @ X])
            model = sm.OLS(y.flatten(), sm.add_constant(X_aug.T))
            results = model.fit()
            
            return results
        
        diffusion_results = estimate_diffusion_params(tfp_matrix.values,
                                                    tfp_matrix.shift().values,
                                                    spatial_weights.values)
        
        # Calculate spillover effects
        direct_effects = diffusion_results.params[1:len(tfp_matrix)+1]
        indirect_effects = diffusion_results.params[len(tfp_matrix)+1:]
        
        # Test for significance of spillovers
        def test_spillovers(results, R):
            """Test for significant spillover effects"""
            wald = results.wald_test(R)
            return {
                'statistic': wald.statistic,
                'p_value': wald.pvalue,
                'df': wald.df_denom
            }
        
        # Construct restriction matrix for spillover test
        R = np.zeros((len(indirect_effects), len(diffusion_results.params)))
        R[:, len(tfp_matrix)+1:] = np.eye(len(indirect_effects))
        spillover_test = test_spillovers(diffusion_results, R)
        
        return {
            'direct_effects': pd.DataFrame(direct_effects, index=tfp_matrix.index),
            'indirect_effects': pd.DataFrame(indirect_effects, index=tfp_matrix.index),
            'spillovers': pd.DataFrame(spatial_weights.values * indirect_effects.mean(),
                                     index=spatial_weights.index,
                                     columns=spatial_weights.columns),
            'spillover_test': spillover_test
        }
    
    def _calculate_input_efficiency(self, output: str, inputs: List[str],
                                  entity_id: str, time_id: str) -> Dict[str, pd.DataFrame]:
        """Calculate input-specific efficiency measures"""
        results = {}
        
        for input_var in inputs:
            # Calculate partial productivity
            partial_prod = np.log(self.data[output]) - np.log(self.data[input_var])
            
            # Estimate input-specific frontier
            def estimate_frontier(y, x):
                # Prepare data for quantile regression
                model = sm.QuantReg(y, sm.add_constant(x))
                
                # Estimate frontier (95th percentile)
                results = model.fit(q=0.95)
                
                return results
            
            frontier_results = estimate_frontier(partial_prod,
                                              np.log(self.data[inputs]))
            
            # Calculate efficiency scores
            efficiency = partial_prod - frontier_results.predict(
                sm.add_constant(np.log(self.data[inputs]))
            )
            
            # Decompose efficiency changes
            efficiency_matrix = efficiency.unstack(time_id)
            tech_change = efficiency_matrix.diff(axis=1).mean()
            catch_up = efficiency_matrix.diff(axis=0).mean()
            
            results[input_var] = pd.DataFrame({
                'partial_productivity': partial_prod,
                'efficiency_score': efficiency,
                'technical_change': tech_change,
                'catch_up_effect': catch_up
            })
        
        return results
    
    def _construct_spillover_network(self, spillover_matrix: pd.DataFrame) -> nx.DiGraph:
        """Construct directed network of technology spillovers"""
        # Create network
        G = nx.from_pandas_adjacency(spillover_matrix, create_using=nx.DiGraph)
        
        # Calculate network metrics
        centrality = nx.eigenvector_centrality_numpy(G)
        clustering = nx.clustering(G)
        
        # Add node attributes
        nx.set_node_attributes(G, centrality, 'centrality')
        nx.set_node_attributes(G, clustering, 'clustering')
        
        return G
    
    def _decompose_variance_components(self, scale_effects: pd.DataFrame,
                                     diffusion: Dict[str, pd.DataFrame],
                                     input_efficiency: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Decompose variance into components"""
        components = {}
        
        # Scale effects variance
        components['scale'] = scale_effects['scale_elasticity'].var()
        
        # Diffusion variance
        components['direct_diffusion'] = diffusion['direct_effects'].stack().var()
        components['indirect_diffusion'] = diffusion['indirect_effects'].stack().var()
        
        # Input-specific variance
        for input_var, results in input_efficiency.items():
            components[f'{input_var}_efficiency'] = results['efficiency_score'].var()
        
        # Calculate proportions
        total_var = sum(components.values())
        proportions = {k: v/total_var for k, v in components.items()}
        
        return pd.DataFrame({
            'variance': components,
            'proportion': proportions
        })
    
    def _calculate_diagnostics(self) -> Dict[str, float]:
        """Calculate model diagnostics"""
        return {
            'log_likelihood': self.results.get('log_likelihood', None),
            'aic': self.results.get('aic', None),
            'bic': self.results.get('bic', None)
        }
    
    def analyze_knowledge_spillovers(self, output: str, inputs: List[str],
                                   rd_expenditure: str, patents: str,
                                   entity_id: str, time_id: str) -> pd.DataFrame:
        """Analyze knowledge spillovers and R&D effects"""
        # Calculate knowledge stocks
        def perpetual_inventory(flows: pd.Series, depreciation: float = 0.15) -> pd.Series:
            stocks = pd.Series(index=flows.index, dtype=float)
            stocks.iloc[0] = flows.iloc[0] / (depreciation + 0.05)  # Initial stock
            for t in range(1, len(flows)):
                stocks.iloc[t] = flows.iloc[t] + (1 - depreciation) * stocks.iloc[t-1]
            return stocks
        
        # Calculate R&D stocks
        rd_stocks = self.data.groupby(entity_id)[rd_expenditure].apply(perpetual_inventory)
        
        # Patent-based knowledge flows
        patent_citations = self.data.groupby([entity_id, time_id])[patents].sum()
        citation_matrix = patent_citations.unstack(entity_id)
        
        # Calculate technological proximity
        def tech_proximity(patents_matrix: pd.DataFrame) -> pd.DataFrame:
            patent_vectors = patents_matrix.fillna(0).values
            proximity = 1 - cdist(patent_vectors.T, patent_vectors.T, metric='cosine')
            return pd.DataFrame(proximity, index=patents_matrix.columns,
                              columns=patents_matrix.columns)
        
        tech_prox = tech_proximity(citation_matrix)
        
        # Estimate knowledge production function
        def estimate_knowledge_production(rd_stock: pd.Series, spillovers: pd.Series,
                                       output: pd.Series) -> Dict:
            X = pd.DataFrame({
                'rd_stock': np.log(rd_stock),
                'spillovers': np.log(spillovers),
                'output': np.log(output)
            })
            X = sm.add_constant(X)
            model = sm.OLS(np.log(self.data[patents]), X)
            results = model.fit()
            return {
                'elasticities': results.params,
                'std_errors': results.bse,
                'r_squared': results.rsquared,
                'diagnostics': {
                    'f_stat': results.fvalue,
                    'p_value': results.f_pvalue
                }
            }
        
        # Calculate inter-industry spillovers
        spillovers = tech_prox @ rd_stocks
        
        # Estimate production function with spillovers
        kpf_results = estimate_knowledge_production(rd_stocks, spillovers,
                                                  self.data[output])
        
        return pd.DataFrame({
            'rd_stocks': rd_stocks,
            'spillovers': spillovers,
            'elasticity_rd': kpf_results['elasticities']['rd_stock'],
            'elasticity_spillover': kpf_results['elasticities']['spillovers'],
            'tech_proximity': tech_prox.mean(axis=1)
        })
    
    def estimate_learning_effects(self, output: str, inputs: List[str],
                                experience: str, entity_id: str,
                                time_id: str) -> Dict[str, pd.DataFrame]:
        """Estimate learning effects and experience accumulation"""
        # Calculate cumulative experience
        cum_exp = self.data.groupby(entity_id)[experience].cumsum()
        
        # Estimate learning curve
        def estimate_learning_curve(y: pd.Series, experience: pd.Series,
                                 controls: pd.DataFrame) -> Dict:
            X = pd.DataFrame({
                'experience': np.log(experience),
                **{f'control_{i}': np.log(controls[col])
                   for i, col in enumerate(controls.columns)}
            })
            X = sm.add_constant(X)
            
            # Estimate with time fixed effects
            time_dummies = pd.get_dummies(self.data[time_id], prefix='time')
            X = pd.concat([X, time_dummies], axis=1)
            
            model = sm.OLS(np.log(y), X)
            results = model.fit()
            
            return {
                'learning_rate': 1 - 2**results.params['experience'],
                'coefficients': results.params,
                'std_errors': results.bse,
                'r_squared': results.rsquared
            }
        
        # Estimate for each input
        learning_results = {}
        for input_var in inputs:
            results = estimate_learning_curve(
                self.data[input_var],
                cum_exp,
                self.data[inputs].drop(input_var, axis=1)
            )
            learning_results[input_var] = pd.DataFrame({
                'learning_rate': results['learning_rate'],
                'experience_elasticity': results['coefficients']['experience'],
                'std_error': results['std_errors']['experience'],
                'r_squared': results['r_squared']
            })
        
        return learning_results
    
    def analyze_technology_gaps(self, output: str, inputs: List[str],
                              entity_id: str, time_id: str) -> pd.DataFrame:
        """Analyze technology gaps and convergence patterns"""
        # Calculate technology frontier
        def estimate_frontier(data: pd.DataFrame, output: str,
                            inputs: List[str]) -> pd.Series:
            log_y = np.log(data[output])
            log_X = np.log(data[inputs])
            
            # Estimate stochastic frontier
            from statsmodels.stats.outliers_influence import OLSInfluence
            
            model = sm.OLS(log_y, sm.add_constant(log_X))
            results = model.fit()
            
            # Identify frontier observations
            influence = OLSInfluence(results)
            frontier_mask = influence.resid_studentized > 0
            
            return data[frontier_mask]
        
        # Calculate frontier by period
        frontiers = self.data.groupby(time_id).apply(
            lambda x: estimate_frontier(x, output, inputs)
        )
        
        # Calculate technology gaps
        gaps = pd.DataFrame(index=self.data.index)
        for period in self.data[time_id].unique():
            period_data = self.data[self.data[time_id] == period]
            frontier_data = frontiers.loc[period]
            
            # Calculate distance to frontier
            gaps.loc[period_data.index, 'tech_gap'] = (
                np.log(frontier_data[output].mean()) -
                np.log(period_data[output])
            )
        
        # Analyze convergence
        def estimate_convergence(gaps: pd.Series, initial_output: pd.Series) -> Dict:
            X = sm.add_constant(np.log(initial_output))
            model = sm.OLS(gaps, X)
            results = model.fit()
            
            return {
                'beta': results.params[1],
                'std_error': results.bse[1],
                'half_life': -np.log(2) / results.params[1],
                'r_squared': results.rsquared
            }
        
        # Estimate convergence by period
        convergence_results = []
        for period in self.data[time_id].unique()[1:]:
            period_gaps = gaps.loc[self.data[time_id] == period, 'tech_gap']
            initial_output = self.data.loc[
                self.data[time_id] == period-1, output
            ]
            
            results = estimate_convergence(period_gaps, initial_output)
            convergence_results.append({
                'period': period,
                **results
            })
        
        convergence_df = pd.DataFrame(convergence_results)
        
        # Combine results
        return pd.DataFrame({
            'tech_gap': gaps['tech_gap'],
            'convergence_rate': convergence_df['beta'].mean(),
            'half_life': convergence_df['half_life'].mean(),
            'r_squared': convergence_df['r_squared'].mean()
        })
    
    def _calculate_diagnostics(self) -> Dict[str, float]:
        """Calculate comprehensive model diagnostics"""
        diagnostics = super()._calculate_diagnostics()
        
        # Add additional diagnostics
        if hasattr(self, 'model') and self.model is not None:
            # Information criteria
            diagnostics.update({
                'aic': self.model.aic,
                'bic': self.model.bic,
                'hqic': self.model.hqic
            })
            
            # Specification tests
            diagnostics['reset_test'] = sm.stats.diagnostic.reset_ramsey(
                self.model, power=3
            ).pvalue
            
            # Heteroskedasticity tests
            diagnostics['breusch_pagan'] = sm.stats.diagnostic.het_breuschpagan(
                self.model.resid, self.model.model.exog
            )[1]
            
            # Serial correlation tests
            diagnostics['durbin_watson'] = sm.stats.stattools.durbin_watson(
                self.model.resid
            )
            
            # Normality tests
            diagnostics['jarque_bera'] = sm.stats.diagnostic.jarque_bera(
                self.model.resid
            )[1]
        
        return diagnostics
