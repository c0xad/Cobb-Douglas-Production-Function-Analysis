import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from advanced_production import AdvancedProductionFunctions
from tfp_analysis import TFPDecomposition, EndogenousGrowthModel
from hierarchical_models import HierarchicalResults
import seaborn as sns
from typing import Dict
import logging
import wbgapi as wb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, countries: list, start_year: int, end_year: int):
        self.countries = countries
        self.start_year = start_year
        self.end_year = end_year
        self.indicators = {
            'gdp': 'NY.GDP.MKTP.KD',
            'labor': 'SL.TLF.TOTL.IN',
            'capital': 'NE.GDI.FTOT.KD',
            'rnd': 'GB.XPD.RSDV.GD.ZS',
            'human_capital': 'SE.TER.ENRR'
        }
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch and process World Bank data"""
        data_frames = []
        
        for indicator_name, indicator_code in self.indicators.items():
            try:
                # Fetch data using wbgapi
                df = wb.data.DataFrame(
                    indicator_code,
                    self.countries,
                    time=range(self.start_year, self.end_year + 1),
                    labels=True
                )
                
                # Clean and reshape data
                df = df.reset_index()
                df = df.melt(
                    id_vars=['time'],
                    var_name='country',
                    value_name=indicator_name
                )
                data_frames.append(df)
                
            except Exception as e:
                logger.error(f"Error fetching {indicator_name}: {str(e)}")
                continue
        
        if not data_frames:
            raise ValueError("No data could be fetched")
        
        # Merge all indicators
        final_data = data_frames[0]
        for df in data_frames[1:]:
            final_data = final_data.merge(
                df, on=['time', 'country'], how='outer'
            )
        
        # Clean and process
        final_data = final_data.sort_values(['country', 'time'])
        final_data = final_data.dropna()
        
        return final_data

def generate_synthetic_data(n_samples: int = 100, n_sectors: int = 3) -> pd.DataFrame:
    """Generate synthetic data if real data fetching fails"""
    np.random.seed(42)
    
    time = np.arange(n_samples)
    sectors = [f"Sector_{i}" for i in range(n_sectors)]
    
    data = []
    for sector in sectors:
        # Base productivity trends
        a_trend = 0.02 + np.random.normal(0, 0.005)
        
        # Generate inputs
        labor = np.exp(np.random.normal(0, 0.3, n_samples) + 0.015 * time)
        capital = np.exp(np.random.normal(0, 0.25, n_samples) + 0.02 * time)
        rnd_expenditure = np.exp(np.random.normal(-1, 0.5, n_samples) + 0.03 * time)
        human_capital = np.exp(np.random.normal(0, 0.2, n_samples) + 0.01 * time)
        
        # Technology level with R&D effects
        technology = np.exp(a_trend * time + 0.1 * np.log(rnd_expenditure))
        
        # Generate output
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

def run_data_collection():
    """Run data collection with fallback to synthetic data"""
    logger.info("Starting data collection...")
    
    try:
        # Try to fetch real data
        countries = ['USA', 'JPN', 'DEU', 'GBR', 'FRA']
        fetcher = DataFetcher(countries, 2000, 2022)
        data = fetcher.fetch_data()
        logger.info("Successfully fetched real data")
        return data
    
    except Exception as e:
        logger.warning(f"Failed to fetch real data: {str(e)}")
        logger.info("Falling back to synthetic data")
        return generate_synthetic_data()

def run_analyses(data: pd.DataFrame) -> Dict:
    """Run all analyses"""
    results = {}
    
    # Production function analysis
    prod_analyzer = AdvancedProductionFunctions(data)
    results['production'] = {
        'leontief': prod_analyzer.estimate_leontief('output', 'labor', 'capital'),
        'ces': prod_analyzer.estimate_dynamic_ces('output', 'labor', 'capital', 'time',
                                                sectors=data['sector'].unique().tolist())
    }
    
    # TFP analysis
    tfp_analyzer = TFPDecomposition(data)
    results['tfp'] = {
        'components': tfp_analyzer.malmquist_index('output', ['labor', 'capital'], 'time')
    }
    
    # Growth analysis
    growth_analyzer = EndogenousGrowthModel(data)
    results['growth'] = {
        'rnd_model': growth_analyzer.estimate_rnd_model('output', 'labor', 'capital',
                                                      'rnd_expenditure', 'human_capital'),
        'convergence': growth_analyzer.analyze_convergence('output_per_capita', 'output',
                                                         'human_capital', 'rnd_intensity')
    }
    
    # Hierarchical analysis
    hierarchical_analyzer = HierarchicalResults(data)
    results['hierarchical'] = {
        'sector_effects': hierarchical_analyzer.estimate_sector_effects('output', ['labor', 'capital', 'rnd_expenditure']),
        'time_effects': hierarchical_analyzer.estimate_time_effects('output_per_capita')
    }
    
    return results

def create_visualizations(data: pd.DataFrame, results: Dict):
    """Create and save all visualizations"""
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # Sector comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Output per capita trends
    plt.subplot(2, 2, 1)
    sns.lineplot(data=data, x='time', y='output_per_capita', hue='sector')
    plt.title('Output per Capita by Sector')
    
    # Plot 2: R&D Intensity
    plt.subplot(2, 2, 2)
    sns.boxplot(data=data, x='sector', y='rnd_intensity')
    plt.title('R&D Intensity Distribution')
    
    # Plot 3: Technology Levels
    plt.subplot(2, 2, 3)
    sns.lineplot(data=data, x='time', y='technology', hue='sector')
    plt.title('Technology Level Evolution')
    
    # Plot 4: Human Capital vs R&D
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=data, x='human_capital', y='rnd_expenditure', hue='sector')
    plt.title('Human Capital vs R&D Investment')
    
    plt.tight_layout()
    plt.savefig('output/sector_analysis.png')
    plt.close()

    # Create hierarchical analysis plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sector Effects
    plt.subplot(1, 2, 1)
    sector_effects = results['hierarchical']['sector_effects']
    sns.barplot(x=sector_effects.index, y=sector_effects.values)
    plt.title('Sector-Specific Effects')
    plt.xticks(rotation=45)
    
    # Plot 2: Time Effects
    plt.subplot(1, 2, 2)
    time_effects = results['hierarchical']['time_effects']
    sns.lineplot(x=time_effects.index, y=time_effects.values)
    plt.title('Time Effects on Productivity')
    
    plt.tight_layout()
    plt.savefig('output/hierarchical_analysis.png')
    plt.close()

def print_summary(results: Dict):
    """Print summary of analysis results"""
    print("\nEconomic Analysis Summary")
    print("========================")
    
    print("\n1. Production Function Analysis")
    print("-----------------------------")
    print(f"Leontief Technical Coefficients:")
    print(results['production']['leontief']['parameters'])
    
    print("\n2. TFP Analysis")
    print("-------------")
    components = results['tfp']['components']
    print(f"Mean Technical Change: {np.mean(components.technical_change):.3f}")
    print(f"Mean Efficiency Change: {np.mean(components.efficiency_change):.3f}")
    
    print("\n3. Growth Analysis")
    print("----------------")
    print(f"Innovation Elasticity: {results['growth']['rnd_model']['innovation_elasticity']:.3f}")
    print(f"Convergence Rate: {results['growth']['convergence']['beta_convergence']:.3f}")
    
    print("\n4. Hierarchical Analysis")
    print("----------------------")
    print("Sector Effects Range:")
    sector_effects = results['hierarchical']['sector_effects']
    print(f"Min: {sector_effects.min():.3f}, Max: {sector_effects.max():.3f}")
    print(f"Time Effects Trend: {results['hierarchical']['time_effects'].mean():.3f} (average annual effect)")

def main():
    """Main execution function"""
    try:
        # Data collection
        data = run_data_collection()
        
        # Run analyses
        results = run_analyses(data)
        
        # Create visualizations
        create_visualizations(data, results)
        
        # Print summary
        print_summary(results)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()