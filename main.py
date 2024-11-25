import numpy as np
import pandas as pd
from cobb_douglas import CobbDouglasAnalysis

def main():
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate time periods
    time = np.arange(1, n_samples + 1)
    
    # Generate labor and capital with some correlation and trend
    labor = np.exp(np.random.normal(0, 0.5, n_samples) + 0.02 * time)
    capital = np.exp(np.random.normal(0, 0.4, n_samples) + 0.015 * time)
    
    # Generate output using Cobb-Douglas production function
    # Y = A * L^α * K^β * exp(ε)
    A = 1.5  # technology parameter
    alpha = 0.6  # labor elasticity
    beta = 0.4   # capital elasticity
    epsilon = np.random.normal(0, 0.1, n_samples)  # random noise
    
    output = A * (labor ** alpha) * (capital ** beta) * np.exp(epsilon)
    
    # Initialize the analysis
    cd_analysis = CobbDouglasAnalysis()
    
    # Load the data
    cd_analysis.load_data(output=output, labor=labor, capital=capital, time=time)
    
    # Estimate parameters using different methods
    print("\nEstimating parameters using different methods:")
    methods = ['ols', 'robust', 'maximum_likelihood']
    for method in methods:
        print(f"\n{method.upper()} Estimation:")
        params = cd_analysis.estimate_parameters(method=method)
        print(f"Alpha (labor): {params['alpha']:.3f}")
        print(f"Beta (capital): {params['beta']:.3f}")
        if 'A' in params:
            print(f"Technology (A): {params['A']:.3f}")
    
    # Perform bootstrap analysis
    print("\nPerforming bootstrap analysis...")
    confidence_intervals = cd_analysis.bootstrap_parameters(n_iterations=1000)
    print("\nBootstrap Confidence Intervals:")
    for param, intervals in confidence_intervals.items():
        print(f"{param}: [{intervals['lower']:.3f}, {intervals['upper']:.3f}]")
    
    # Analyze returns to scale
    print("\nAnalyzing returns to scale...")
    rts_analysis = cd_analysis.analyze_returns_to_scale()
    print(f"Regime: {rts_analysis['regime']}")
    print(f"Scale Elasticity: {rts_analysis['scale_elasticity']:.3f}")
    print(f"Interpretation: {rts_analysis['interpretation']}")
    
    # Estimate technical efficiency
    print("\nEstimating technical efficiency...")
    efficiency_results = cd_analysis.estimate_technical_efficiency()
    print(f"Mean Technical Efficiency: {efficiency_results['mean_efficiency']:.3f}")
    print(f"Median Technical Efficiency: {efficiency_results['median_efficiency']:.3f}")
    
    # Calculate marginal products
    print("\nCalculating marginal products and elasticities...")
    mp_results = cd_analysis.calculate_marginal_products()
    print("\nMean Marginal Products:")
    print(f"Labor: {mp_results['marginal_products']['MP_L'].mean():.3f}")
    print(f"Capital: {mp_results['marginal_products']['MP_K'].mean():.3f}")
    
    # Perform diagnostic tests
    print("\nPerforming diagnostic tests...")
    diagnostics = cd_analysis.diagnostic_tests()
    print("\nDiagnostic Test Results:")
    print(f"Heteroskedasticity p-value: {diagnostics['heteroskedasticity']['p_value']:.3f}")
    print(f"Durbin-Watson statistic: {diagnostics['autocorrelation']['durbin_watson']:.3f}")
    print(f"Normality test p-value: {diagnostics['normality']['p_value']:.3f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    cd_analysis.visualize_data()
    cd_analysis.plot_production_function()
    
    print("\nAnalysis complete! Check the visualization_output directory for plots.")

if __name__ == "__main__":
    main() 