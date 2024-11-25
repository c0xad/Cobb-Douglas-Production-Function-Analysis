import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TFPComponents:
    technical_change: np.ndarray
    efficiency_change: np.ndarray
    scale_efficiency: np.ndarray
    total_tfp: np.ndarray

class TFPDecomposition:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize TFP decomposition analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing output, inputs, and time variables
        """
        self.data = data
        self.results = None
        
    def malmquist_index(self, output: str, inputs: List[str], time_var: str) -> TFPComponents:
        """
        Calculate Malmquist Productivity Index and decompose TFP
        
        Parameters:
        -----------
        output : str
            Name of output variable
        inputs : List[str]
            Names of input variables
        time_var : str
            Name of time variable
        """
        def distance_function(x: np.ndarray, y: float, frontier_data: pd.DataFrame) -> float:
            # Calculate distance to frontier using DEA
            n_inputs = len(x)
            n_dmu = len(frontier_data)
            
            # Linear programming for efficiency score
            def objective(lambda_weights):
                return -np.sum(lambda_weights * frontier_data[output])
            
            constraints = []
            # Input constraints
            for i in range(n_inputs):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda l, i=i: np.sum(l * frontier_data[inputs[i]]) - x[i]
                })
            
            # Output constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda l: np.sum(l * frontier_data[output]) - y
            })
            
            # Convexity constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda l: np.sum(l) - 1
            })
            
            res = minimize(
                objective,
                x0=np.ones(n_dmu)/n_dmu,
                constraints=constraints,
                method='SLSQP'
            )
            
            return -res.fun if res.success else np.nan
        
        # Calculate components for each period
        n_periods = len(self.data[time_var].unique())
        technical_change = np.zeros(n_periods - 1)
        efficiency_change = np.zeros(n_periods - 1)
        scale_efficiency = np.zeros(n_periods - 1)
        
        for t in range(n_periods - 1):
            period_t = self.data[self.data[time_var] == t]
            period_t1 = self.data[self.data[time_var] == t + 1]
            
            # Calculate distance functions
            d_t_t = np.array([distance_function(row[inputs].values, row[output],
                                              period_t) for _, row in period_t.iterrows()])
            d_t1_t1 = np.array([distance_function(row[inputs].values, row[output],
                                                period_t1) for _, row in period_t1.iterrows()])
            d_t_t1 = np.array([distance_function(row[inputs].values, row[output],
                                               period_t) for _, row in period_t1.iterrows()])
            d_t1_t = np.array([distance_function(row[inputs].values, row[output],
                                               period_t1) for _, row in period_t.iterrows()])
            
            # Calculate components
            efficiency_change[t] = np.mean(d_t1_t1 / d_t_t)
            technical_change[t] = np.mean(np.sqrt((d_t_t / d_t1_t) * (d_t_t1 / d_t1_t1)))
            
            # Calculate scale efficiency
            scale_t = np.mean(d_t_t) / np.mean(d_t1_t)
            scale_t1 = np.mean(d_t_t1) / np.mean(d_t1_t1)
            scale_efficiency[t] = np.sqrt(scale_t * scale_t1)
        
        total_tfp = technical_change * efficiency_change * scale_efficiency
        
        return TFPComponents(
            technical_change=technical_change,
            efficiency_change=efficiency_change,
            scale_efficiency=scale_efficiency,
            total_tfp=total_tfp
        )
    
    def visualize_decomposition(self, components: TFPComponents, time_var: str):
        """
        Create visualization of TFP components
        """
        periods = self.data[time_var].unique()[:-1]  # Exclude last period
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot components
        ax1.plot(periods, components.technical_change, label='Technical Change', marker='o')
        ax1.plot(periods, components.efficiency_change, label='Efficiency Change', marker='s')
        ax1.plot(periods, components.scale_efficiency, label='Scale Efficiency', marker='^')
        ax1.set_title('TFP Components Over Time')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Index')
        ax1.legend()
        ax1.grid(True)
        
        # Plot total TFP
        ax2.plot(periods, components.total_tfp, label='Total TFP', marker='o', color='red')
        ax2.set_title('Total Factor Productivity Over Time')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Index')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

class EndogenousGrowthModel:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize endogenous growth model analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing output, inputs, and growth factors
        """
        self.data = data
        self.results = None
        
    def estimate_rnd_model(self, output: str, labor: str, capital: str,
                          rnd: str, human_capital: str) -> Dict:
        """
        Estimate endogenous growth model with R&D and human capital
        
        Parameters:
        -----------
        output : str
            Name of output variable
        labor : str
            Name of labor input variable
        capital : str
            Name of capital input variable
        rnd : str
            Name of R&D expenditure variable
        human_capital : str
            Name of human capital variable
        """
        # Take logs of variables
        log_y = np.log(self.data[output])
        log_l = np.log(self.data[labor])
        log_k = np.log(self.data[capital])
        log_rnd = np.log(self.data[rnd])
        log_h = np.log(self.data[human_capital])
        
        # Create interaction terms
        rnd_hc = log_rnd * log_h
        
        # Prepare data for regression
        X = pd.DataFrame({
            'log_l': log_l,
            'log_k': log_k,
            'log_rnd': log_rnd,
            'log_h': log_h,
            'rnd_hc': rnd_hc
        })
        X = sm.add_constant(X)
        
        # Estimate model
        model = sm.OLS(log_y, X)
        results = model.fit()
        
        # Calculate growth contributions
        contributions = {
            'labor': results.params['log_l'] * np.mean(np.diff(log_l)),
            'capital': results.params['log_k'] * np.mean(np.diff(log_k)),
            'rnd': results.params['log_rnd'] * np.mean(np.diff(log_rnd)),
            'human_capital': results.params['log_h'] * np.mean(np.diff(log_h)),
            'interaction': results.params['rnd_hc'] * np.mean(np.diff(rnd_hc))
        }
        
        # Calculate innovation elasticity
        innovation_elasticity = (results.params['log_rnd'] +
                               results.params['rnd_hc'] * np.mean(log_h))
        
        return {
            'model_results': results,
            'contributions': contributions,
            'innovation_elasticity': innovation_elasticity
        }
    
    def analyze_convergence(self, output_per_capita: str, initial_output: str,
                          human_capital: str, rnd_intensity: str) -> Dict:
        """
        Analyze convergence patterns in growth
        
        Parameters:
        -----------
        output_per_capita : str
            Name of output per capita variable
        initial_output : str
            Name of initial output variable
        human_capital : str
            Name of human capital variable
        rnd_intensity : str
            Name of R&D intensity variable
        """
        # Calculate growth rate
        growth_rate = np.log(self.data[output_per_capita]).diff()
        
        # Prepare data for convergence regression
        X = pd.DataFrame({
            'initial_output': np.log(self.data[initial_output]),
            'human_capital': self.data[human_capital],
            'rnd_intensity': self.data[rnd_intensity]
        })
        X = sm.add_constant(X)
        
        # Estimate convergence model
        model = sm.OLS(growth_rate.dropna(), X.iloc[1:])
        results = model.fit()
        
        # Calculate convergence rate
        beta_convergence = -results.params['initial_output']
        half_life = np.log(2) / beta_convergence if beta_convergence > 0 else np.inf
        
        # Test for conditional convergence
        conditional_factors = {
            'human_capital': results.params['human_capital'],
            'rnd_intensity': results.params['rnd_intensity']
        }
        
        return {
            'model_results': results,
            'beta_convergence': beta_convergence,
            'half_life': half_life,
            'conditional_factors': conditional_factors
        }
    
    def visualize_growth_patterns(self, output_per_capita: str, rnd_intensity: str,
                                human_capital: str, time_var: str):
        """
        Create visualizations of growth patterns and relationships
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Growth path
        axes[0, 0].plot(self.data[time_var], np.log(self.data[output_per_capita]),
                       marker='o')
        axes[0, 0].set_title('Growth Path')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Log Output per Capita')
        
        # R&D Intensity vs Growth
        growth_rate = np.log(self.data[output_per_capita]).diff()
        axes[0, 1].scatter(self.data[rnd_intensity].iloc[1:], growth_rate.dropna())
        axes[0, 1].set_title('R&D Intensity vs Growth')
        axes[0, 1].set_xlabel('R&D Intensity')
        axes[0, 1].set_ylabel('Growth Rate')
        
        # Human Capital vs Growth
        axes[1, 0].scatter(self.data[human_capital].iloc[1:], growth_rate.dropna())
        axes[1, 0].set_title('Human Capital vs Growth')
        axes[1, 0].set_xlabel('Human Capital')
        axes[1, 0].set_ylabel('Growth Rate')
        
        # Innovation System
        axes[1, 1].scatter(self.data[rnd_intensity], self.data[human_capital])
        axes[1, 1].set_title('Innovation System')
        axes[1, 1].set_xlabel('R&D Intensity')
        axes[1, 1].set_ylabel('Human Capital')
        
        plt.tight_layout()
        return fig 