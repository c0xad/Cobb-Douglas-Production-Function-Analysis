import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import chi2
import warnings
import optuna
from pathlib import Path

class CobbDouglasAnalysis:
    def __init__(self):
        self.model = None
        self.params = {}
        self.data = None
        self.bootstrap_results = None
        self.efficiency_scores = None
        self.frontier_params = None
        self.visualization_dir = Path('visualization_output')
        self.visualization_dir.mkdir(exist_ok=True)
        
    def load_data(self, output, labor, capital, time=None):
        """
        Load and prepare data for analysis
        
        Parameters:
        -----------
        output : array-like
            Production output (Q)
        labor : array-like
            Labor input (L)
        capital : array-like
            Capital input (K)
        time : array-like, optional
            Time periods for panel data analysis
        """
        self.data = pd.DataFrame({
            'Q': output,
            'L': labor,
            'K': capital,
            'ln_Q': np.log(output),
            'ln_L': np.log(labor),
            'ln_K': np.log(capital)
        })
        
        if time is not None:
            self.data['time'] = time
            self.data['ln_time'] = np.log(time)
    
    def estimate_parameters(self, method='ols', specification='cobb_douglas'):
        """
        Estimate production function parameters using various methods and specifications
        
        Parameters:
        -----------
        method : str
            Estimation method ('ols', 'robust', 'maximum_likelihood', 'gmm')
        specification : str
            Function specification ('cobb_douglas', 'translog', 'ces')
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
            
        if specification == 'cobb_douglas':
            X = self.data[['ln_L', 'ln_K']]
            y = self.data['ln_Q']
        elif specification == 'translog':
            # Create translog terms
            self.data['ln_L_sq'] = self.data['ln_L'] ** 2
            self.data['ln_K_sq'] = self.data['ln_K'] ** 2
            self.data['ln_L_K'] = self.data['ln_L'] * self.data['ln_K']
            X = self.data[['ln_L', 'ln_K', 'ln_L_sq', 'ln_K_sq', 'ln_L_K']]
            y = self.data['ln_Q']
        elif specification == 'ces':
            # Use hyperparameter tuning for CES
            return self.ces_hyperparameter_tuning()
        
        if method == 'gmm':
            # Generalized Method of Moments estimation
            def gmm_objective(params):
                beta = params
                residuals = y - X @ beta
                moment_conditions = X.T @ residuals
                return moment_conditions.T @ moment_conditions

            result = minimize(gmm_objective, x0=np.zeros(X.shape[1]))
            self.params['coefficients'] = result.x
            
            # Calculate GMM standard errors
            n = len(y)
            residuals = y - X @ result.x
            S = (X.T @ X) / n
            Omega = np.zeros((X.shape[1], X.shape[1]))
            for i in range(n):
                moment = X.iloc[i:i+1].T @ residuals[i:i+1]
                Omega += moment @ moment.T
            Omega = Omega / n
            V = np.linalg.inv(S) @ Omega @ np.linalg.inv(S) / n
            self.params['gmm_std_errors'] = np.sqrt(np.diag(V))
            
        elif method == 'ols':
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            # Store parameters
            self.params['alpha'] = self.model.coef_[0]
            self.params['beta'] = self.model.coef_[1]
            self.params['A'] = np.exp(self.model.intercept_)
            
            # Calculate standard errors and t-statistics
            n = len(y)
            k = X.shape[1]
            y_pred = self.model.predict(X)
            mse = np.sum((y - y_pred) ** 2) / (n - k - 1)
            var_coef = mse * np.linalg.inv(X.T @ X)
            
            self.params['std_errors'] = np.sqrt(np.diag(var_coef))
            self.params['t_stats'] = self.model.coef_ / self.params['std_errors']
            self.params['p_values'] = 2 * (1 - stats.t.cdf(np.abs(self.params['t_stats']), n - k - 1))
            
        elif method == 'robust':
            model = sm.RLM(y, sm.add_constant(X))
            results = model.fit()
            
            self.params['alpha'] = results.params.iloc[1]
            self.params['beta'] = results.params.iloc[2]
            self.params['A'] = np.exp(results.params.iloc[0])
            self.params['std_errors'] = results.bse.iloc[1:]
            self.params['t_stats'] = results.tvalues.iloc[1:]
            self.params['p_values'] = results.pvalues.iloc[1:]
            
        elif method == 'maximum_likelihood':
            # Maximum likelihood estimation with technical efficiency
            def neg_log_likelihood(params):
                alpha, beta, sigma = params
                predicted = alpha * self.data['ln_L'] + beta * self.data['ln_K']
                return -np.sum(norm.logpdf(self.data['ln_Q'] - predicted, scale=sigma))
            
            result = minimize(neg_log_likelihood, x0=[0.3, 0.3, 1.0],
                            bounds=[(0, 1), (0, 1), (0.001, None)])
            
            self.params['alpha'] = result.x[0]
            self.params['beta'] = result.x[1]
            self.params['sigma'] = result.x[2]
        
        # Calculate R-squared and adjusted R-squared
        if method != 'ces':  # Skip for CES since it uses a different structure
            self.params['r2'] = self.model.score(X, y)
            n = len(y)
            k = X.shape[1]
            self.params['adj_r2'] = 1 - (1 - self.params['r2']) * (n - 1) / (n - k - 1)
        
        return self.params

    def ces_hyperparameter_tuning(self):
        """
        Use Optuna for hyperparameter tuning of the CES production function
        """
        def objective(trial):
            rho = trial.suggest_float('rho', -0.99, 0.99)
            delta = trial.suggest_float('delta', 0.01, 0.99)
            A = trial.suggest_float('A', 0.01, 10.0)
            
            if rho == 0:  # Handle limiting case (Cobb-Douglas)
                predicted = np.log(A) + delta * np.log(self.data['L']) + (1-delta) * np.log(self.data['K'])
            else:
                predicted = np.log(A) + (1/rho) * np.log(delta * self.data['L']**(-rho) + (1-delta) * self.data['K']**(-rho))
            
            loss = np.sum((self.data['ln_Q'] - predicted) ** 2)
            return loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        
        self.params['rho'] = best_params['rho']
        self.params['delta'] = best_params['delta']
        self.params['A'] = best_params['A']
        self.params['elasticity_substitution'] = 1 / (1 + self.params['rho'])
        
        return self.params

    def dynamic_returns_to_scale(self):
        """
        Analyze dynamic returns to scale with time-varying parameters
        """
        if 'time' not in self.data.columns:
            raise ValueError("Time series data required for dynamic analysis")

        results = []
        window_size = min(20, len(self.data) // 4)  # Adaptive window size

        for i in range(len(self.data) - window_size + 1):
            window_data = self.data.iloc[i:i+window_size]
            X = window_data[['ln_L', 'ln_K']]
            y = window_data['ln_Q']
            
            model = LinearRegression()
            model.fit(X, y)
            
            results.append({
                'time': self.data['time'].iloc[i+window_size-1],
                'alpha': model.coef_[0],
                'beta': model.coef_[1],
                'returns_to_scale': sum(model.coef_)
            })
        
        dynamic_results = pd.DataFrame(results)
        
        # Test for structural breaks
        def chow_test(data, breakpoint):
            before = data[:breakpoint]
            after = data[breakpoint:]
            
            # Full model
            X_full = self.data[['ln_L', 'ln_K']]
            y_full = self.data['ln_Q']
            model_full = LinearRegression().fit(X_full, y_full)
            rss_full = np.sum((y_full - model_full.predict(X_full))**2)
            
            # Split models
            X_before = before[['ln_L', 'ln_K']]
            y_before = before['ln_Q']
            model_before = LinearRegression().fit(X_before, y_before)
            rss_before = np.sum((y_before - model_before.predict(X_before))**2)
            
            X_after = after[['ln_L', 'ln_K']]
            y_after = after['ln_Q']
            model_after = LinearRegression().fit(X_after, y_after)
            rss_after = np.sum((y_after - model_after.predict(X_after))**2)
            
            # Calculate F-statistic
            n = len(self.data)
            k = 2  # number of parameters
            f_stat = ((rss_full - (rss_before + rss_after)) / k) / ((rss_before + rss_after) / (n - 2*k))
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)
            
            return f_stat, p_value

        # Test for structural breaks at multiple points
        breakpoints = []
        for i in range(window_size, len(self.data) - window_size):
            f_stat, p_value = chow_test(self.data, i)
            if p_value < 0.05:
                breakpoints.append({
                    'time': self.data['time'].iloc[i],
                    'f_stat': f_stat,
                    'p_value': p_value
                })

        return {
            'dynamic_parameters': dynamic_results,
            'structural_breaks': pd.DataFrame(breakpoints) if breakpoints else None
        }

    def advanced_diagnostic_tests(self):
        """
        Perform advanced diagnostic tests including:
        - Ramsey RESET test for functional form
        - Ljung-Box test for autocorrelation
        - ARCH test for heteroskedasticity
        - Stationarity tests
        """
        if self.data is None or not self.params:
            raise ValueError("Data must be loaded and parameters estimated first")

        X = self.data[['ln_L', 'ln_K']]
        y = self.data['ln_Q']
        residuals = y - self.model.predict(X)

        # Ramsey RESET test
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        yhat = model.predict(X_with_const)
        X_reset = sm.add_constant(np.column_stack((X, yhat**2, yhat**3)))
        model_reset = sm.OLS(y, X_reset).fit()
        reset_f_stat = ((model.ssr - model_reset.ssr)/2) / (model_reset.ssr/(len(y)-5))
        reset_p_value = 1 - stats.f.cdf(reset_f_stat, 2, len(y)-5)

        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)

        # ARCH test
        resid_sq = residuals**2
        X_arch = sm.add_constant(pd.concat([resid_sq.shift(i) for i in range(1, 5)], axis=1).dropna())
        y_arch = resid_sq[4:]
        model_arch = sm.OLS(y_arch, X_arch).fit()
        arch_stat = len(y_arch) * model_arch.rsquared
        arch_p_value = 1 - chi2.cdf(arch_stat, 4)

        # Stationarity tests
        adf_results = {}
        kpss_results = {}
        for var in ['ln_Q', 'ln_L', 'ln_K']:
            adf_results[var] = adfuller(self.data[var], regression='ct')
            kpss_results[var] = kpss(self.data[var], regression='ct', nlags='auto')

        return {
            'ramsey_reset': {
                'f_statistic': reset_f_stat,
                'p_value': reset_p_value
            },
            'ljung_box': lb_test,
            'arch_test': {
                'statistic': arch_stat,
                'p_value': arch_p_value
            },
            'stationarity': {
                'adf_results': adf_results,
                'kpss_results': kpss_results
            }
        }

    def plot_advanced_diagnostics(self):
        """
        Create advanced diagnostic plots
        """
        if self.data is None or not self.params:
            raise ValueError("Data must be loaded and parameters estimated first")

        fig = plt.figure(figsize=(20, 20))

        # 1. Parameter stability plot
        if hasattr(self, 'dynamic_results'):
            ax1 = plt.subplot(3, 2, 1)
            self.dynamic_results.plot(x='time', y=['alpha', 'beta', 'returns_to_scale'], ax=ax1)
            ax1.set_title('Parameter Stability Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Parameter Values')

        # 2. CUSUM test plot
        ax2 = plt.subplot(3, 2, 2)
        residuals = self.data['ln_Q'] - self.model.predict(self.data[['ln_L', 'ln_K']])
        cusum = np.cumsum(residuals) / np.std(residuals)
        ax2.plot(range(len(cusum)), cusum)
        critical_value = 1.96 * np.sqrt(np.arange(1, len(cusum) + 1))
        ax2.plot(range(len(cusum)), critical_value, 'r--')
        ax2.plot(range(len(cusum)), -critical_value, 'r--')
        ax2.set_title('CUSUM Test')

        # 3. Partial regression plots
        fig = sm.graphics.plot_partregress_grid(
            sm.OLS(self.data['ln_Q'], sm.add_constant(self.data[['ln_L', 'ln_K']])).fit(),
            fig=fig
        )

        # 4. Added variable plots
        sm.graphics.plot_regress_exog(
            sm.OLS(self.data['ln_Q'], sm.add_constant(self.data[['ln_L', 'ln_K']])).fit(),
            'ln_L',
            fig=fig
        )

        # 5. Influence plots
        fig = sm.graphics.influence_plot(
            sm.OLS(self.data['ln_Q'], sm.add_constant(self.data[['ln_L', 'ln_K']])).fit(),
            criterion='cooks'
        )

        plt.tight_layout()
        return fig

    def estimate_productivity_components(self):
        """
        Decompose total factor productivity into technical change, efficiency change,
        and scale efficiency change using the Malmquist productivity index
        """
        if 'time' not in self.data.columns:
            raise ValueError("Time series data required for productivity decomposition")

        results = []
        for t in range(1, len(self.data)):
            # Technical efficiency in periods t and t-1
            te_t = self.efficiency_scores.iloc[t]
            te_t1 = self.efficiency_scores.iloc[t-1]
            
            # Technical change
            tech_change = np.exp(self.model.predict(self.data[['ln_L', 'ln_K']].iloc[t:t+1]) -
                               self.model.predict(self.data[['ln_L', 'ln_K']].iloc[t-1:t]))
            
            # Efficiency change
            eff_change = te_t / te_t1
            
            # Scale efficiency change
            returns_t = self.params['alpha'] + self.params['beta']
            scale_eff = (self.data['Q'].iloc[t] / self.data['Q'].iloc[t-1]) ** (1 - returns_t)
            
            results.append({
                'time': self.data['time'].iloc[t],
                'technical_change': tech_change,
                'efficiency_change': eff_change,
                'scale_efficiency_change': scale_eff,
                'tfp_change': tech_change * eff_change * scale_eff
            })
        
        return pd.DataFrame(results)
    
    def bootstrap_parameters(self, n_iterations=1000):
        """
        Perform bootstrap analysis to estimate parameter confidence intervals
        """
        if self.data is None or not self.params:
            raise ValueError("Data must be loaded and parameters estimated first")
        
        bootstrap_params = []
        X = self.data[['ln_L', 'ln_K']]
        y = self.data['ln_Q']
        n_samples = len(y)
        
        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Fit model
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            
            bootstrap_params.append({
                'alpha': model.coef_[0],
                'beta': model.coef_[1],
                'A': np.exp(model.intercept_)
            })
        
        self.bootstrap_results = pd.DataFrame(bootstrap_params)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for param in ['alpha', 'beta', 'A']:
            confidence_intervals[param] = {
                'lower': np.percentile(self.bootstrap_results[param], 2.5),
                'upper': np.percentile(self.bootstrap_results[param], 97.5)
            }
        
        return confidence_intervals
    
    def estimate_technical_efficiency(self):
        """
        Estimate technical efficiency using stochastic frontier analysis
        """
        if self.data is None or not self.params:
            raise ValueError("Data must be loaded and parameters estimated first")
        
        # Calculate predicted output
        ln_Q_pred = (self.params['alpha'] * self.data['ln_L'] + 
                    self.params['beta'] * self.data['ln_K'])
        
        # Calculate residuals
        residuals = self.data['ln_Q'] - ln_Q_pred
        
        # Estimate technical efficiency as exp(-u_i)
        # where u_i is the non-negative technical inefficiency term
        self.efficiency_scores = np.exp(-np.maximum(0, -residuals))
        
        return {
            'efficiency_scores': self.efficiency_scores,
            'mean_efficiency': np.mean(self.efficiency_scores),
            'median_efficiency': np.median(self.efficiency_scores)
        }
    
    def estimate_production_frontier(self):
        """
        Estimate the production frontier using corrected ordinary least squares
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        X = self.data[['ln_L', 'ln_K']]
        y = self.data['ln_Q']
        
        # Step 1: OLS estimation
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)
        
        # Step 2: Correct intercept
        max_residual = np.max(residuals)
        corrected_intercept = model.intercept_ + max_residual
        
        self.frontier_params = {
            'alpha': model.coef_[0],
            'beta': model.coef_[1],
            'A': np.exp(corrected_intercept)
        }
        
        return self.frontier_params
    
    def diagnostic_tests(self):
        """
        Perform diagnostic tests on the model
        """
        if self.data is None or not self.params:
            raise ValueError("Data must be loaded and parameters estimated first")
        
        X = sm.add_constant(self.data[['ln_L', 'ln_K']])
        y = self.data['ln_Q']
        
        # Breusch-Pagan test for heteroskedasticity
        bp_test = het_breuschpagan(self.model.predict(self.data[['ln_L', 'ln_K']]) - y, X)
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(self.model.predict(self.data[['ln_L', 'ln_K']]) - y)
        
        # Jarque-Bera test for normality
        residuals = self.model.predict(self.data[['ln_L', 'ln_K']]) - y
        jb_stat = stats.jarque_bera(residuals)
        
        return {
            'heteroskedasticity': {
                'statistic': bp_test[0],
                'p_value': bp_test[1]
            },
            'autocorrelation': {
                'durbin_watson': dw_stat
            },
            'normality': {
                'jarque_bera_stat': jb_stat[0],
                'p_value': jb_stat[1]
            }
        }
    
    def calculate_elasticity_of_substitution(self):
        """
        Calculate elasticity of substitution between labor and capital
        """
        if not self.params:
            raise ValueError("Parameters must be estimated first")
        
        # Calculate elasticity of substitution (σ) for Cobb-Douglas
        # For Cobb-Douglas, σ is always 1, but we'll calculate related metrics
        self.data['K_L_ratio'] = self.data['K'] / self.data['L']
        self.data['MP_ratio'] = (self.params['beta'] * self.data['L']) / (self.params['alpha'] * self.data['K'])
        
        return {
            'elasticity': 1.0,
            'K_L_ratio': self.data['K_L_ratio'],
            'MP_ratio': self.data['MP_ratio']
        }
    
    def calculate_marginal_products(self):
        """
        Calculate marginal products and elasticities
        """
        if not self.params:
            raise ValueError("Parameters must be estimated first")
            
        self.data['MP_L'] = self.params['alpha'] * (self.data['Q'] / self.data['L'])
        self.data['MP_K'] = self.params['beta'] * (self.data['Q'] / self.data['K'])
        
        # Calculate elasticities
        self.data['output_elasticity_L'] = self.params['alpha']
        self.data['output_elasticity_K'] = self.params['beta']
        
        # Calculate marginal rate of technical substitution (MRTS)
        self.data['MRTS'] = self.data['MP_L'] / self.data['MP_K']
        
        return {
            'marginal_products': self.data[['MP_L', 'MP_K']],
            'elasticities': {
                'labor': self.params['alpha'],
                'capital': self.params['beta']
            },
            'MRTS': self.data['MRTS']
        }
    
    def analyze_returns_to_scale(self):
        """
        Comprehensive analysis of returns to scale
        """
        if not self.params:
            raise ValueError("Parameters must be estimated first")
            
        returns_sum = self.params['alpha'] + self.params['beta']
        
        # Calculate scale elasticity
        scale_elasticity = returns_sum
        
        # Determine returns to scale regime
        if np.isclose(returns_sum, 1, rtol=1e-2):
            regime = "Constant Returns to Scale"
        elif returns_sum > 1:
            regime = "Increasing Returns to Scale"
        else:
            regime = "Decreasing Returns to Scale"
            
        # Calculate optimal scale
        if 'time' in self.data.columns:
            # Time-varying returns to scale
            self.data['scale_efficiency'] = (self.data['Q'] / 
                np.sqrt(self.data['L'] * self.data['K']))
        
        return {
            'regime': regime,
            'scale_elasticity': scale_elasticity,
            'returns_sum': returns_sum,
            'interpretation': f"A 1% increase in all inputs leads to a {returns_sum:.1%} increase in output"
        }
    
    def plot_production_function(self):
        """
        Create comprehensive visualizations of the production function
        """
        if self.data is None or not self.params:
            raise ValueError("Data must be loaded and parameters estimated first")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Actual vs. Predicted Output
        ax1 = plt.subplot(2, 2, 1)
        predicted_Q = np.exp(self.model.predict(self.data[['ln_L', 'ln_K']]))
        ax1.scatter(self.data['Q'], predicted_Q, alpha=0.5)
        ax1.plot([self.data['Q'].min(), self.data['Q'].max()],
                [self.data['Q'].min(), self.data['Q'].max()],
                'r--', label='Perfect Prediction')
        ax1.set_xlabel('Actual Output')
        ax1.set_ylabel('Predicted Output')
        ax1.set_title('Actual vs. Predicted Production')
        ax1.legend()
        
        # 2. 3D Production Surface
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        L_grid = np.linspace(self.data['L'].min(), self.data['L'].max(), 50)
        K_grid = np.linspace(self.data['K'].min(), self.data['K'].max(), 50)
        L_mesh, K_mesh = np.meshgrid(L_grid, K_grid)
        Q_mesh = self.params['A'] * (L_mesh ** self.params['alpha']) * (K_mesh ** self.params['beta'])
        
        surf = ax2.plot_surface(np.log(L_mesh), np.log(K_mesh), np.log(Q_mesh),
                              cmap='viridis', alpha=0.8)
        ax2.scatter(self.data['ln_L'], self.data['ln_K'], self.data['ln_Q'],
                   c='red', marker='o')
        ax2.set_xlabel('ln(Labor)')
        ax2.set_ylabel('ln(Capital)')
        ax2.set_zlabel('ln(Output)')
        ax2.set_title('Production Surface')
        plt.colorbar(surf, ax=ax2, label='ln(Output)')
        
        # 3. Residual Analysis
        ax3 = plt.subplot(2, 2, 3)
        residuals = self.data['ln_Q'] - self.model.predict(self.data[['ln_L', 'ln_K']])
        sm.graphics.qqplot(residuals, line='45', fit=True, ax=ax3)
        ax3.set_title('Q-Q Plot of Residuals')
        
        # 4. Technical Efficiency Distribution
        if self.efficiency_scores is not None:
            ax4 = plt.subplot(2, 2, 4)
            sns.histplot(self.efficiency_scores, kde=True, ax=ax4)
            ax4.axvline(np.mean(self.efficiency_scores), color='r', linestyle='--',
                       label=f'Mean Efficiency: {np.mean(self.efficiency_scores):.3f}')
            ax4.set_title('Technical Efficiency Distribution')
            ax4.set_xlabel('Technical Efficiency Score')
            ax4.legend()
        
        plt.tight_layout()
        return fig

    def visualize_data(self):
        """Generate comprehensive visualizations of the input data"""
        # Set up the plotting style
        plt.style.use('default')  # Using default matplotlib style instead of seaborn
        
        # Time series plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        for i, var in enumerate(['ln_Q', 'ln_L', 'ln_K']):
            axes[i].plot(self.data['time'], self.data[var], label=var)
            axes[i].set_title(f'Time Series of {var}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(var)
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'time_series.png')
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = self.data[['ln_Q', 'ln_L', 'ln_K']].corr()
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(3), ['ln_Q', 'ln_L', 'ln_K'])
        plt.yticks(range(3), ['ln_Q', 'ln_L', 'ln_K'])
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center')
        plt.title('Correlation Heatmap')
        plt.savefig(self.visualization_dir / 'correlation_heatmap.png')
        plt.close()
        
        # Scatter plots with regression lines
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Output vs Labor
        ax1.scatter(self.data['ln_L'], self.data['ln_Q'], alpha=0.5)
        z = np.polyfit(self.data['ln_L'], self.data['ln_Q'], 1)
        p = np.poly1d(z)
        ax1.plot(self.data['ln_L'], p(self.data['ln_L']), "r--", alpha=0.8)
        ax1.set_xlabel('ln(Labor)')
        ax1.set_ylabel('ln(Output)')
        ax1.set_title('Output vs Labor')
        
        # Output vs Capital
        ax2.scatter(self.data['ln_K'], self.data['ln_Q'], alpha=0.5)
        z = np.polyfit(self.data['ln_K'], self.data['ln_Q'], 1)
        p = np.poly1d(z)
        ax2.plot(self.data['ln_K'], p(self.data['ln_K']), "r--", alpha=0.8)
        ax2.set_xlabel('ln(Capital)')
        ax2.set_ylabel('ln(Output)')
        ax2.set_title('Output vs Capital')
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'factor_relationships.png')
        plt.close()

    def estimate(self):
        """Estimate the Cobb-Douglas production function parameters"""
        # Prepare the model
        X = sm.add_constant(self.data[['ln_L', 'ln_K', 'ln_L_squared', 'ln_K_squared', 'ln_LK_interaction']])
        y = self.data['ln_Q']
        
        # Estimate the model
        model = sm.OLS(y, X)
        self.results = model.fit()
        
        # Generate detailed report
        report = []
        report.append("# Cobb-Douglas Production Function Analysis Report\n")
        
        # Model summary
        report.append("## Model Summary\n")
        report.append("### Basic Statistics\n")
        report.append(f"- R-squared: {self.results.rsquared:.4f}\n")
        report.append(f"- Adjusted R-squared: {self.results.rsquared_adj:.4f}\n")
        report.append(f"- F-statistic: {self.results.fvalue:.4f}\n")
        report.append(f"- Prob (F-statistic): {self.results.f_pvalue:.4f}\n")
        
        # Parameter estimates
        report.append("\n### Parameter Estimates\n")
        params = self.results.params
        conf_int = self.results.conf_int()
        for var in params.index:
            report.append(f"- {var}:")
            report.append(f"  - Coefficient: {params[var]:.4f}")
            report.append(f"  - 95% CI: [{conf_int.loc[var][0]:.4f}, {conf_int.loc[var][1]:.4f}]")
            report.append(f"  - P-value: {self.results.pvalues[var]:.4f}\n")
        
        # Returns to scale analysis
        returns = params['ln_L'] + params['ln_K']
        report.append("\n## Returns to Scale Analysis\n")
        report.append(f"- Returns to scale: {returns:.4f}\n")
        report.append(f"- Type: {'Increasing' if returns > 1 else 'Decreasing' if returns < 1 else 'Constant'} returns to scale\n")
        
        # Save the report
        with open('cobb_douglas_analysis.md', 'w') as f:
            f.write('\n'.join(report))
        
        return self.results
    
    def plot_residuals(self):
        """Generate diagnostic plots for the residuals"""
        if self.results is None:
            raise ValueError("Must run estimate() before plotting residuals")
        
        # QQ plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sm.graphics.qqplot(self.results.resid, dist=stats.norm, line='45', ax=ax1)
        ax1.set_title('Q-Q Plot of Residuals')
        
        # Residuals vs Fitted
        sns.scatterplot(x=self.results.fittedvalues, y=self.results.resid, ax=ax2)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Residuals vs Fitted Values')
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'residual_diagnostics.png')
        plt.close()
