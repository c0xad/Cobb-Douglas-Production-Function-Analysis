import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from advanced_production import AdvancedProductionFunctions
from tfp_analysis import TFPDecomposition, EndogenousGrowthModel
import seaborn as sns
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 100, n_sectors: int = 3) -> pd.DataFrame:
    """Generate synthetic data for economic analysis"""
    np.random.seed(42)
    
    # Time periods
    time = np.arange(n_samples)
    sectors = [f"Sector_{i}" for i in range(n_sectors)]
    
    data = []
    for sector in sectors:
        # Base productivity trends
        a_trend = 0.02 + np.random.normal(0, 0.005)
        
        # Generate inputs with sector-specific trends
        labor = np.exp(np.random.normal(0, 0.3, n_samples) + 0.015 * time)
        capital = np.exp(np.random.normal(0, 0.25, n_samples) + 0.02 * time)
        
        # R&D and human capital components
        rnd_expenditure = np.exp(np.random.normal(-1, 0.5, n_samples) + 0.03 * time)
        human_capital = np.exp(np.random.normal(0, 0.2, n_samples) + 0.01 * time)
        
        # Technology level with R&D effects
        technology = np.exp(a_trend * time + 0.1 * np.log(rnd_expenditure))
        
        # Generate output with all components
        output = (technology * 
                 (labor ** 0.4) * 
                 (capital ** 0.35) * 
                 (human_capital ** 0.15) * 
                 np.exp(np.random.normal(0, 0.05, n_samples)))
        
        # Calculate derived metrics
        rnd_intensity = rnd_expenditure / output
        output_per_capita = output / labor
        
        for t in range(n_samples):
            data.append({
                'time': t,
                'sector': sector,
                'output': output[t],
                'labor': labor[t],
                'capital': capital[t],
                'rnd_expenditure': rnd_expenditure[t],
                'human_capital': human_capital[t],
                'rnd_intensity': rnd_intensity[t],
                'output_per_capita': output_per_capita[t],
                'technology': technology[t]
            })
    
    return pd.DataFrame(data)

def run_production_analysis(data: pd.DataFrame) -> Dict:
    """Run comprehensive production function analysis"""
    logger.info("Starting production function analysis...")
    
    try:
        # Initialize production function analyzer
        prod_analyzer = AdvancedProductionFunctions(data)
        
        # Estimate Leontief production function
        leontief_results = prod_analyzer.estimate_leontief('output', 'labor', 'capital')
        
        # Ensure data is valid for CES estimation
        data_valid = data[['output', 'labor', 'capital', 'time']].notna().all().all()
        if not data_valid:
            raise ValueError("Missing values detected in input data")
        
        # Estimate dynamic CES production function
        ces_results = prod_analyzer.estimate_dynamic_ces(
            'output', 'labor', 'capital', 'time',
            sectors=data['sector'].unique().tolist()
        )
        
        # Calculate marginal products
        mp_results = prod_analyzer.calculate_marginal_products(
            'output', 'labor', 'capital',
            model_type='ces', time_var='time'
        )
        
        # Test returns to scale
        rts_results = prod_analyzer.test_returns_to_scale(
            'output', 'labor', 'capital',
            model_type='ces', time_var='time'
        )
        
        return {
            'leontief': leontief_results,
            'ces': ces_results,
            'marginal_products': mp_results,
            'returns_to_scale': rts_results
        }
    except Exception as e:
        logger.error(f"Error in production analysis: {str(e)}")
        return {
            'error': str(e)
        }

def run_tfp_analysis(data: pd.DataFrame) -> Dict:
    """Run TFP decomposition analysis"""
    logger.info("Starting TFP decomposition analysis...")
    
    try:
        # Initialize TFP analyzer
        tfp_analyzer = TFPDecomposition(data)
        
        # Calculate Malmquist productivity index
        tfp_components = tfp_analyzer.malmquist_index(
            'output',
            ['labor', 'capital'],
            'time'
        )
        
        # Visualize decomposition
        tfp_plot = tfp_analyzer.visualize_decomposition(tfp_components, 'time')
        plt.savefig('tfp_decomposition.png')
        plt.close()
        
        return {
            'components': tfp_components,
            'plot_saved': 'tfp_decomposition.png'
        }
    except Exception as e:
        logger.error(f"Error in TFP analysis: {str(e)}")
        return {
            'error': str(e)
        }

def run_growth_analysis(data: pd.DataFrame) -> Dict:
    """Run endogenous growth analysis"""
    logger.info("Starting endogenous growth analysis...")
    
    try:
        # Initialize growth model analyzer
        growth_analyzer = EndogenousGrowthModel(data)
        
        # Estimate R&D model
        rnd_results = growth_analyzer.estimate_rnd_model(
            'output',
            'labor',
            'capital',
            'rnd_expenditure',
            'human_capital'
        )
        
        # Analyze convergence
        convergence_results = growth_analyzer.analyze_convergence(
            'output_per_capita',
            'output',
            'human_capital',
            'rnd_intensity'
        )
        
        # Visualize growth patterns
        growth_plot = growth_analyzer.visualize_growth_patterns(
            'output_per_capita',
            'rnd_intensity',
            'human_capital',
            'time'
        )
        plt.savefig('growth_patterns.png')
        plt.close()
        
        return {
            'rnd_model': rnd_results,
            'convergence': convergence_results,
            'plot_saved': 'growth_patterns.png'
        }
    except Exception as e:
        logger.error(f"Error in growth analysis: {str(e)}")
        return {
            'error': str(e)
        }

def visualize_sector_comparison(data: pd.DataFrame):
    """Create comparative visualizations across sectors"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Output per capita trends
    plt.subplot(2, 2, 1)
    for sector in data['sector'].unique():
        sector_data = data[data['sector'] == sector]
        plt.plot(sector_data['time'], sector_data['output_per_capita'], label=sector)
    plt.title('Output per Capita by Sector')
    plt.xlabel('Time')
    plt.ylabel('Output per Capita')
    plt.legend()
    
    # Plot 2: R&D Intensity
    plt.subplot(2, 2, 2)
    sns.boxplot(data=data, x='sector', y='rnd_intensity')
    plt.title('R&D Intensity Distribution by Sector')
    plt.xticks(rotation=45)
    
    # Plot 3: Technology Levels
    plt.subplot(2, 2, 3)
    for sector in data['sector'].unique():
        sector_data = data[data['sector'] == sector]
        plt.plot(sector_data['time'], sector_data['technology'], label=sector)
    plt.title('Technology Level by Sector')
    plt.xlabel('Time')
    plt.ylabel('Technology Level')
    plt.legend()
    
    # Plot 4: Human Capital vs R&D
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=data, x='human_capital', y='rnd_expenditure', hue='sector')
    plt.title('Human Capital vs R&D Expenditure')
    
    plt.tight_layout()
    plt.savefig('sector_comparison.png')
    plt.close()

def main():
    """Main execution function"""
    logger.info("Starting economic analysis...")
    
    try:
        # Generate synthetic data
        data = generate_synthetic_data(n_samples=100, n_sectors=3)
        logger.info(f"Generated synthetic data with shape: {data.shape}")
        
        # Run all analyses
        production_results = run_production_analysis(data)
        if 'error' in production_results:
            logger.warning(f"Production analysis failed: {production_results['error']}")
        
        tfp_results = run_tfp_analysis(data)
        if 'error' in tfp_results:
            logger.warning(f"TFP analysis failed: {tfp_results['error']}")
        
        growth_results = run_growth_analysis(data)
        if 'error' in growth_results:
            logger.warning(f"Growth analysis failed: {growth_results['error']}")
        
        # Create sector comparisons
        try:
            visualize_sector_comparison(data)
        except Exception as e:
            logger.warning(f"Sector comparison visualization failed: {str(e)}")
        
        # Print summary results
        print("\nSummary of Analysis Results:")
        print("===========================")
        
        if 'error' not in production_results:
            print("\n1. Production Function Analysis:")
            print(f"- Leontief Technical Coefficients: {production_results['leontief']['parameters']}")
            print(f"- CES Mean Elasticity: {np.mean([params['elasticity_mean'] for params in production_results['ces'].values()]):.3f}")
        
        if 'error' not in tfp_results:
            print("\n2. TFP Analysis:")
            print(f"- Mean Technical Change: {np.mean(tfp_results['components'].technical_change):.3f}")
            print(f"- Mean Efficiency Change: {np.mean(tfp_results['components'].efficiency_change):.3f}")
        
        if 'error' not in growth_results:
            print("\n3. Growth Analysis:")
            print(f"- Innovation Elasticity: {growth_results['rnd_model']['innovation_elasticity']:.3f}")
            print(f"- Convergence Rate: {growth_results['convergence']['beta_convergence']:.3f}")
        
        logger.info("Analysis complete. Check the output directory for visualization plots.")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 