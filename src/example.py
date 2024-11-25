import numpy as np
from cobb_douglas import CobbDouglasAnalysis

# Generate sample data
np.random.seed(42)
n_samples = 100

# True parameters
A_true = 1.5
alpha_true = 0.6
beta_true = 0.4

# Generate inputs
L = np.random.lognormal(mean=2, sigma=0.5, size=n_samples)
K = np.random.lognormal(mean=3, sigma=0.7, size=n_samples)

# Generate output with some noise
Q = A_true * (L ** alpha_true) * (K ** beta_true) * np.random.lognormal(mean=0, sigma=0.1, size=n_samples)

# Create and use the Cobb-Douglas analysis
cd = CobbDouglasAnalysis()

# Load the data
cd.load_data(Q, L, K)

# Estimate parameters
params = cd.estimate_parameters()
print("\nEstimated Parameters:")
print(f"Technology (A): {params['A']:.3f}")
print(f"Labor Elasticity (α): {params['alpha']:.3f}")
print(f"Capital Elasticity (β): {params['beta']:.3f}")
print(f"R-squared: {params['r2']:.3f}")

# Analyze returns to scale
rts = cd.analyze_returns_to_scale()
print(f"\nReturns to Scale: {rts}")

# Calculate and display marginal products
mp = cd.calculate_marginal_products()
print("\nMarginal Products (mean values):")
print(f"Labor: {mp['MP_L'].mean():.3f}")
print(f"Capital: {mp['MP_K'].mean():.3f}")

# Calculate and display average products
ap = cd.calculate_average_products()
print("\nAverage Products (mean values):")
print(f"Labor: {ap['AP_L'].mean():.3f}")
print(f"Capital: {ap['AP_K'].mean():.3f}")

# Plot the results
fig = cd.plot_production_function()
fig.savefig('production_function_analysis.png')
print("\nPlot saved as 'production_function_analysis.png'") 