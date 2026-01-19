# Portfolio Simulator

A Streamlit-based GUI for backtesting portfolio strategies using event-driven rebalancing.

## 🚀 Live Demo

Access the app at: `https://your-app-name.streamlit.app` *(update this after deployment)*

## Overview

This application allows you to:

1. **Load Data** - Import preprocessed `.pkl` files containing stock selection data
2. **Build Events** - Configure how stocks are selected and weighted at each rebalancing event
3. **Run Simulations** - Backtest your strategy with configurable parameters
4. **Analyze Results** - View performance metrics, interactive charts, and holdings heatmaps
5. **Compare Runs** - Overlay multiple simulation runs and export configurations

## Features

- 📊 **Interactive Plotly visualizations** with hover tooltips
- 🔥 **Holdings heatmap** showing per-stock returns by selection rank
- 📈 **Benchmark comparison** (SPY, NDX, GLD, or custom)
- 💰 **Flexible cash policies** (fixed amount or percentage)
- ⚖️ **Multiple weighting schemes** (equal, proportional, softmax, inverse rank)
- 🔀 **Ensemble simulations** with event shuffling
- 📉 **Comprehensive performance metrics** (CAGR, Sharpe, Sortino, Max Drawdown, etc.)

## Installation (Local Development)

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-simulator.git
cd portfolio-simulator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run portfolio_gui_app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
portfolio-simulator/
├── portfolio_gui_app.py      # Main Streamlit application
├── portfolio_sim.py          # Core simulation engine
├── portfolio_gui_core.py     # Event building & simulation orchestration
├── portfolio_gui_io.py       # File I/O and data extraction
├── portfolio_gui_plotting.py # Plotly visualization utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Input Data Format

The app expects a `.pkl` (pickle) file containing stock selection data. Supported formats:

| Format | Description |
|--------|-------------|
| `list[DataFrame]` | List of pandas DataFrames |
| `dict[str, DataFrame]` | Dictionary mapping sheet names to DataFrames |
| `dict` with `sheets`, `sheet_dfs`, `dfs`, or `data` key | Wrapper containing one of the above |

Each DataFrame should contain at minimum:
- **Ticker column** - Stock symbols (Yahoo Finance format, e.g., `AAPL`, `MSFT`)
- **Datetime column** - Event/rebalancing dates
- **Ordering metric column** - Numeric values used to rank and select stocks

### Example DataFrame Structure

| datetime | yf_Ticker | score | other_metric |
|----------|-----------|-------|--------------|
| 2024-01-15 | AAPL | 0.85 | 12.3 |
| 2024-01-15 | MSFT | 0.72 | 8.7 |
| 2024-01-15 | GOOGL | 0.68 | 15.2 |
| 2024-02-15 | NVDA | 0.91 | 22.1 |
| ... | ... | ... | ... |

## Usage Guide

### Step 1: Load & Validate
- Upload your `.pkl` file
- Select which sheets to include and their order
- Map columns (ticker, datetime, ordering metric)

### Step 2: Build Events
- Choose selection direction (ascending/descending)
- Set top-N stocks per event
- Configure weighting scheme
- Optionally include/exclude specific tickers

### Step 3: Simulation Parameters
- Set date range and initial capital
- Configure cash policy
- Add benchmarks for comparison
- Enable ensemble mode for robustness testing

### Step 4: Run & Analyze
- Execute the simulation
- View performance summary table
- Explore interactive charts
- Examine the holdings heatmap
- Download trade logs and data

### Step 5: Compare & Export
- Compare multiple simulation runs
- Export configurations for reproducibility

## Security Note

⚠️ **Pickle files can execute arbitrary code.** Only load `.pkl` files that you created yourself or from sources you trust completely.

## Deployment to Streamlit Cloud

1. Push this repository to GitHub (can be private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and `portfolio_gui_app.py`
5. Click "Deploy"

Your app will be available at `https://your-app-name.streamlit.app`

## Configuration Options

### Event Building
| Parameter | Description |
|-----------|-------------|
| Direction | `ascend` (low values = better) or `descend` (high values = better) |
| Top N | Number of stocks to select per event |
| Weight Mode | `equal`, `proportional`, `softmax`, `inverse_rank` |
| Softmax τ | Temperature for softmax weighting (lower = more concentrated) |

### Simulation
| Parameter | Description |
|-----------|-------------|
| Initial Capital | Starting portfolio value |
| Cash Policy | `None`, `fixed`, or `proportion` |
| Shuffle Events | Enable ensemble mode with randomized event timing |
| Max Leverage | Allow borrowing (>1.0 enables margin) |

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add your license here]

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Market data from [yfinance](https://github.com/ranaroussi/yfinance)
- Visualizations powered by [Plotly](https://plotly.com/)
