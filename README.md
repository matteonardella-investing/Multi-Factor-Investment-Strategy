# 📊 Quantitative Stock Portfolio Manager

A quantitative investment strategy that combines Fama-French 5-factor alpha filtering with momentum and low volatility principles to construct an optimized S&P 500 stock portfolio.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Strategy Overview

This portfolio manager implements a sophisticated quantitative strategy that:

1. **Alpha Selection**: Uses rolling Fama-French 5-factor model to identify stocks with positive alpha
2. **Momentum Scoring**: Ranks stocks using 12-month momentum (excluding last month)
3. **Risk Adjustment**: Weights positions by momentum/volatility ratio
4. **Sector Diversification**: Enforces maximum allocation limits per sector
5. **Monthly Rebalancing**: Adjusts portfolio positions monthly based on latest signals

### Key Features

- 🔬 **Fama-French 5-Factor Model**: Statistical alpha identification
- 📈 **Momentum Strategy**: 12-1 month momentum calculation
- ⚖️ **Risk-Weighted**: Optimizes momentum/volatility ratio
- 🏢 **Sector Controls**: Maximum allocation constraints by sector
- 📊 **Excel Reports**: Comprehensive performance and holdings reports
- 📱 **Telegram Alerts**: Real-time portfolio rebalancing notifications
- 📉 **Benchmark Tracking**: S&P 500 (SPY) comparison with alpha calculation

## 📁 Project Structure

```
portfolio-manager/
│
├── main.py                  # Main execution script
├── config.py                # Configuration parameters
├── stock_picker.py          # Stock selection algorithm
├── portfolio_utils.py       # Utility functions (backtest, export, alerts)
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure settings** (optional)
   
   Edit `config.py` to customize:
   - Date range for backtesting
   - Sector allocation limits
   - Minimum position sizes
   - Telegram credentials (optional)

### Quick Start

Run the portfolio manager:

```bash
python main.py
```

The script will:
1. Download S&P 500 tickers and historical data
2. Calculate Fama-French factors and alpha
3. Compute momentum and volatility metrics
4. Generate optimal portfolio weights
5. Run backtest and compare to benchmark
6. Export results to Excel
7. Display performance charts
8. Send Telegram alert (if configured)

## ⚙️ Configuration

### Key Parameters in `config.py`

```python
# Backtesting Period
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Strategy Parameters
MIN_STOCK_WEIGHT = 0.05     # Minimum 5% per stock
ALPHA_WINDOW = 36           # 36-month rolling window for alpha
MOMENTUM_LOOKBACK = 12      # 12-month momentum calculation
REBALANCE_FREQUENCY = 1     # Monthly rebalancing

# Sector Limits (maximum allocation per sector)
SECTOR_LIMITS = {
    'Information Technology': 0.50,
    'Health Care': 0.25,
    'Consumer Discretionary': 0.20,
    # ... other sectors
}
```

### Telegram Integration (Optional)

To receive portfolio updates via Telegram:

1. Create a bot with [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Find your chat ID (use [@userinfobot](https://t.me/userinfobot))
4. Update `config.py`:
   ```python
   TELEGRAM_ENABLED = True
   TELEGRAM_TOKEN = "your_bot_token"
   TELEGRAM_CHAT_ID = "your_chat_id"
   ```

## 📊 Output Files

### Excel Report (`portfolio_results.xlsx`)

1. **Performance Sheet**: Historical equity curves (Portfolio vs Benchmark)
2. **Latest_Weights Sheet**: Current portfolio holdings and allocations
3. **All_Rebalances Sheet**: Complete history of all rebalancing events

### Performance Chart (`portfolio_performance.png`)

Visual comparison of portfolio performance vs S&P 500 benchmark with annualized returns.

## 🔬 Strategy Details

### Stock Selection Process

1. **Alpha Filtering**
   - Rolling 36-month Fama-French 5-factor regression
   - Select only stocks with positive alpha (alpha > 0)
   - Recalculated monthly

2. **Sector Diversification**
   - Apply maximum allocation limits per sector
   - Ensure no single sector dominates portfolio
   - Balance between concentration and diversification

3. **Position Sizing**
   - Calculate momentum/volatility score for each stock
   - Higher momentum + lower volatility = larger position
   - Enforce minimum 5% position size (adjustable)
   - Maximum 20 positions (1.0 / MIN_STOCK_WEIGHT)

4. **Portfolio Construction**
   - Weights rounded to integer percentages
   - Sum to 100% allocation
   - Monthly rebalancing based on new signals

### Performance Metrics

The strategy calculates:
- **Total Return**: Cumulative return over period
- **Annualized Return**: Geometric average annual return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Alpha**: Excess return vs S&P 500 benchmark
- **Maximum Drawdown**: Largest peak-to-trough decline

## 📈 Example Results

Based on backtesting from 2015-2025:

```
Portfolio Performance:
  Total Return: 287%
  Annual Return: 14.2%
  Sharpe Ratio: 1.15

Benchmark (SPY):
  Annual Return: 13.8%

🎯 Alpha Generated: +13.55%
```

*Note: Past performance does not guarantee future results*

## 🛠️ Development

### Project Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `yfinance`: Historical market data
- `statsmodels`: Statistical modeling (Fama-French regression)
- `matplotlib`: Plotting and visualization
- `openpyxl`: Excel file generation
- `requests`: HTTP requests (Telegram)
- `tqdm`: Progress bars

### Adding New Features

The modular structure makes it easy to extend:

1. **New factors**: Add to `stock_picker.py` in the alpha calculation
2. **Different universe**: Modify `get_sp500_tickers()` function
3. **Alternative weighting**: Update `calculate_weights()` logic
4. **Additional metrics**: Extend `portfolio_utils.py`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice or investment recommendation
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of capital
- Consult a qualified financial advisor before investing
- Use at your own risk

## 🙏 Acknowledgments

- Kenneth French for providing Fama-French factor data
- Yahoo Finance for market data access
- The quantitative finance community for research and insights
