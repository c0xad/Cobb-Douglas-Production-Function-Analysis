# Advanced Cobb-Douglas Production Function Analysis

## Overview
This project implements a sophisticated econometric analysis toolkit for estimating and analyzing Cobb-Douglas production functions using real-world economic data. It combines World Bank data integration, advanced statistical methods, and comprehensive visualization capabilities to provide deep insights into production economics.

## Features

### Data Collection and Processing (`data_fetcher.py`)
- **World Bank API Integration**: Seamless fetching of economic indicators through `wbgapi`
- **Key Economic Indicators**:
  - GDP (NY.GDP.MKTP.CN)
  - Total Labor Force (SL.TLF.TOTL.IN)
  - Industry Value Added (NV.IND.TOTL.CN)
  - Total Population (SP.POP.TOTL)
  - Inflation Rates (FP.CPI.TOTL.ZG)
- **Advanced Preprocessing**:
  - Inflation adjustment for real value calculations
  - Time series smoothing with configurable windows
  - Automated outlier detection and handling
  - Comprehensive preprocessing reports

### Production Function Analysis (`cobb_douglas.py`)
- **Multiple Estimation Methods**:
  - Ordinary Least Squares (OLS)
  - Robust Regression (RLM)
  - Maximum Likelihood Estimation (MLE)
  - Stochastic Frontier Analysis
- **Statistical Features**:
  - Returns to scale analysis
  - Technical efficiency estimation
  - Bootstrap confidence intervals
  - Hypothesis testing for constant returns
- **Visualization Suite**:
  - Time series analysis plots
  - Correlation heatmaps
  - Factor relationship scatter plots
  - Residual diagnostics
  - Production frontier visualization

## Technical Details

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
scipy>=1.7.0
wbgapi>=1.0.0
optuna>=2.10.0
scikit-learn>=0.24.0
```

### Architecture
The project follows a modular architecture with two main components:

1. **Data Management Layer** (`data_fetcher.py`):
   - `WorldBankDataFetcher`: Handles data acquisition and preprocessing
   - Implements data quality validation
   - Provides flexible data transformation options

2. **Analysis Layer** (`cobb_douglas.py`):
   - `CobbDouglasAnalysis`: Core analysis functionality
   - Implements multiple estimation strategies
   - Generates comprehensive visualizations and reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cobb-douglas-analysis.git
cd cobb-douglas-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from src.data_fetcher import WorldBankDataFetcher
from src.cobb_douglas import CobbDouglasAnalysis

# Initialize data fetcher
fetcher = WorldBankDataFetcher(country_codes=['USA'], start_year=2000, end_year=2022)

# Fetch and preprocess data
data = fetcher.fetch_data()
processed_data = fetcher.preprocess_for_estimation(data)

# Initialize analysis
analysis = CobbDouglasAnalysis()
analysis.load_data(processed_data)

# Run analysis
analysis.estimate(method='robust')
analysis.visualize_data()
```

### Advanced Usage
```python
# Custom preprocessing options
processed_data = fetcher.preprocess_for_estimation(
    data,
    apply_smoothing=True,
    adjust_for_inflation=True
)

# Advanced estimation with bootstrapping
analysis.estimate(
    method='maximum_likelihood',
    bootstrap_iterations=1000,
    confidence_level=0.95
)

# Generate comprehensive reports
analysis.plot_residuals()
analysis.plot_production_frontier()
```

## Output Files
- `preprocessing_report.md`: Detailed data preprocessing statistics
- `cobb_douglas_analysis.md`: Comprehensive analysis results
- `visualization_output/`: Directory containing all generated plots
  - `time_series.png`: Economic indicator trends
  - `correlation_heatmap.png`: Factor correlation analysis
  - `factor_relationships.png`: Production relationships
  - `residual_diagnostics.png`: Model validation plots

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite:
```bibtex
@software{cobb_douglas_analysis,
  author = {Your Name},
  title = {Advanced Cobb-Douglas Production Function Analysis},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/cobb-douglas-analysis}
}
```

## Acknowledgments
- World Bank for providing the economic data through their API
- The statsmodels team for their comprehensive econometric tools
- The scientific Python community for their excellent libraries
