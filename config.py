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
# TRANSACTION COSTS SETTINGS
# =============================================================================

ENABLE_TRANSACTION_COSTS = True

# Bid-Ask Spread
BID_ASK_SPREAD = 0.0005  # 0.05% per trade

# Commission costs
COMMISSION_PER_TRADE = 1.0     # Fixed $1 per trade
COMMISSION_PER_SHARE = 0.005   # $0.005 per share

# Market impact (scales with trade size)
MARKET_IMPACT_FACTOR = 0.0001  # 1 basis point per 1% of portfolio traded

# Portfolio size for calculating per-trade and per-share commissions
ASSUMED_PORTFOLIO_VALUE = 100000  # $100,000

# =============================================================================
# TURNOVER CONSTRAINTS
# =============================================================================

ENABLE_TURNOVER_CONSTRAINT = True

# Maximum turnover per rebalancing period (50% = half of portfolio can be traded)
# Turnover = sum(|new_weight - old_weight|) / 2
MAX_MONTHLY_TURNOVER = 0.50  # 50% maximum

# Minimum weight change to trigger a trade (helps reduce unnecessary small trades)
MIN_TRADE_THRESHOLD = 0.01  # 1% minimum change to trade

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
# MONTE CARLO SETTINGS
# =============================================================================

MONTE_CARLO_ENABLED = True

# Number of simulated paths (higher = more precise, slower)
MC_N_SIMULATIONS = 10000

# Forward projection horizon in months
MC_HORIZON_MONTHS = 120  # 5 years

# Confidence levels for percentile envelopes
MC_CONFIDENCE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Random seed for reproducibility (set to None for different results each run)
MC_RANDOM_SEED = 42

# Number of sample paths to plot (visual only, doesn't affect stats)
MC_SAMPLE_PATHS_PLOT = 200

# Output files
MC_PLOT_FILE = 'monte_carlo_simulation.png'
MC_EXCEL_FILE = 'monte_carlo_results.xlsx'

# =============================================================================
# ROBUSTNESS MONTE CARLO SETTINGS (Approach 3)
# =============================================================================

ROBUSTNESS_MC_ENABLED = True

# Number of perturbed strategy runs (each reruns weight calc + backtest)
# 200 is a good balance of speed vs precision (~5-15 min depending on machine)
ROBUSTNESS_N_SIMULATIONS = 200

# Noise scale: std of Gaussian noise applied to momentum scores
# 0.20 = 20% noise (moderate test), 0.40 = 40% noise (aggressive test)
ROBUSTNESS_NOISE_SCALE = 0.20

# Random seed for reproducibility
ROBUSTNESS_RANDOM_SEED = 42

# Output files
ROBUSTNESS_PLOT_FILE = 'robustness_monte_carlo.png'
ROBUSTNESS_EXCEL_FILE = 'robustness_monte_carlo_results.xlsx'

# =============================================================================
# TELEGRAM SETTINGS
# =============================================================================

TELEGRAM_ENABLED = True
TELEGRAM_TOKEN = "your_telegram_toker"
TELEGRAM_CHAT_ID = "your_chat_ID"

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

EXCEL_OUTPUT_FILE = 'portfolio_results.xlsx'
PLOT_ENABLED = True
