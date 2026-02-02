# =============================================================================
# PORTFOLIO CONFIGURATION FILE - STOCKS ONLY
# =============================================================================

from datetime import datetime

# =============================================================================
# GENERAL SETTINGS
# =============================================================================

START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

BENCHMARK_TICKER = 'SPY'

# =============================================================================
# REBALANCING SETTINGS
# =============================================================================

REBALANCE_FREQUENCY = 1   # Monthly rebalancing

# =============================================================================
# STOCK STRATEGY SETTINGS
# =============================================================================

# Sector limits (maximum allocation per sector)
# Total sum exceeds 100% because these are maximum limits only
SECTOR_LIMITS = {
    'Information Technology': 0.50,
    'Health Care': 0.25,
    'Consumer Discretionary': 0.20,
    'Financials': 0.15,
    'Industrials': 0.10,
    'Communication Services': 0.10,
    'Consumer Staples': 0.10,
    'Energy': 0.10,
    'Materials': 0.10,
    'Real Estate': 0.10,
    'Utilities': 0.10
}

MIN_STOCK_WEIGHT = 0.05  # Minimum 5% per stock
ALPHA_WINDOW = 36        # Rolling window for alpha calculation (months)
MOMENTUM_LOOKBACK = 12   # Momentum lookback period (months)

# Portfolio construction
ROUND_TO_INTEGER = True  # Round weights to integer percentages (e.g., 7%, 8%, not 7.3%)

# =============================================================================
# TELEGRAM SETTINGS
# =============================================================================

TELEGRAM_ENABLED = False
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

EXCEL_OUTPUT_FILE = 'portfolio_results.xlsx'
PLOT_ENABLED = True
