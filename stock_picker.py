import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import requests
from datetime import datetime
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:.8f}'.format

# =============================================================================
# 1: DOWNLOAD PRICES
# =============================================================================

def download_daily_prices(tickers, start_date, end_date):

    from tqdm import tqdm

    print(f"\n{'=' * 60}")
    print(f"📥 DOWNLOAD PRICES: {len(tickers)} ticker")
    print(f"{'=' * 60}")

    # FAST METHOD: download all tickers at once
    print("\n⚡ Fast batch download in progress...")

    try:
        # yfinance can download multiple tickers simultaneously
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=True,
            auto_adjust=True,
            group_by='ticker',
            threads=True
        )

        # Extract closing prices
        prices = pd.DataFrame()
        failed_tickers = []

        print("\n📊 Processing downloaded data...")
        for ticker in tqdm(tickers, desc="Processing", unit="ticker", ncols=80):
            try:
                if len(tickers) == 1:
                    # Single ticker case
                    if 'Close' in data.columns:
                        prices[ticker] = data['Close']
                    else:
                        failed_tickers.append(ticker)
                else:
                    # Multiple tickers case
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker]
                        if 'Close' in ticker_data.columns:
                            prices[ticker] = ticker_data['Close']
                        else:
                            failed_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)

        print(f"\n✓ Download done:")
        print(f"  - Success: {len(prices.columns)} ticker")
        print(f"  - Failures: {len(failed_tickers)} ticker")

        # Remove tickers with too much missing data (>20%)
        threshold = int(len(prices) * 0.8)
        prices = prices.dropna(axis=1, thresh=threshold)
        print(f"  - After filtering missing data: {len(prices.columns)} ticker")

        return prices, failed_tickers

    except Exception as e:
        print(f"\n❌ Batch download failed: {e}")
        print("⚠️  Falling back to individual downloads...")

        # Fallback: individual download with progress bar
        prices = pd.DataFrame()
        failed_tickers = []

        for ticker in tqdm(tickers, desc="Downloading", unit="ticker", ncols=80):
            try:
                data = yf.download(ticker, start=start_date, end=end_date,
                                   progress=False, auto_adjust=True)

                if not data.empty and 'Close' in data.columns:
                    prices[ticker] = data['Close']
                else:
                    failed_tickers.append(ticker)

            except Exception as e:
                failed_tickers.append(ticker)

        threshold = int(len(prices) * 0.8)
        prices = prices.dropna(axis=1, thresh=threshold)

        return prices, failed_tickers

# =============================================================================
# 2: DOWNLOAD FAMA-FRENCH FACTORS
# =============================================================================

def get_fama_french_factors(start_date, end_date):
    print(f"\n{'=' * 60}")
    print(f"📥 DOWNLOADING FAMA-FRENCH 5 FACTORS")
    print(f"{'=' * 60}")

    try:
        import urllib.request
        import zipfile
        import io

        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'

        with urllib.request.urlopen(url) as response:
            zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
            csv_file = zip_file.namelist()[0]
            with zip_file.open(csv_file) as f:
                df = pd.read_csv(f, skiprows=3)

        df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

        df['Date'] = df['Date'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        df = df / 100
        df = df.loc[start_date:end_date]

        if df.empty:
            print("❌ ATTENTION: DataFrame Fama-French is empty after date filter!")
        else:
            print(f"✓ SUCCESS! RF mean: {df['RF'].mean():.8f}")

        return df

    except Exception as e:
        print(f"❌ Download ERROR: {e}")
        return None

# =============================================================================
# STEP 3: CALCULATE MONTHLY RETURNS
# =============================================================================

def create_monthly_returns(df_daily, ff_factors):
    print(f"\n{'=' * 60}")
    print(f"📊 CALCULATING MONTHLY RETURNS")
    print(f"{'=' * 60}")

    # Monthly stock returns
    monthly_prices = df_daily.resample('ME').last()
    monthly_returns = monthly_prices.pct_change()

    # Monthly RF and Factors
    monthly_ff = ff_factors.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # Separate RF from other factors
    monthly_rf = monthly_ff['RF']
    monthly_factors = monthly_ff[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]

    # =========================================================================
    # FORWARD-FILL: extend FF factors for months where prices exist but
    # Fama-French data is not yet published. Uses last available factors.
    # =========================================================================
    prices_dates = monthly_returns.index
    ff_dates = monthly_ff.index
    
    # Find months in prices that are beyond the last available FF date
    missing_months = prices_dates[prices_dates > ff_dates.max()]
    
    if len(missing_months) > 0:
        last_ff_date = ff_dates.max()
        print(f"  ⚠️  Fama-French factors end at {last_ff_date.strftime('%Y-%m')}")
        print(f"  ⚠️  Prices extend {len(missing_months)} month(s) beyond: "
              f"{', '.join(d.strftime('%Y-%m') for d in missing_months)}")
        print(f"  ➡️  Forward-filling last available factors for missing months")
        
        # Reindex RF and factors to cover all price dates, forward-fill
        monthly_rf = monthly_rf.reindex(prices_dates).ffill()
        monthly_factors = monthly_factors.reindex(prices_dates).ffill()
    
    # Align: use all dates where BOTH returns and (possibly extended) factors exist
    common_dates = monthly_returns.index.intersection(monthly_rf.index)

    monthly_returns = monthly_returns.loc[common_dates]
    monthly_rf = monthly_rf.loc[common_dates]
    monthly_factors = monthly_factors.loc[common_dates]

    # Remove any NaNs originating from the initial pct_change
    valid_mask = monthly_returns.iloc[:, 0].notna()
    monthly_returns = monthly_returns[valid_mask]
    monthly_rf = monthly_rf[valid_mask]
    monthly_factors = monthly_factors[valid_mask]

    # Excess Returns
    excess_returns = monthly_returns.sub(monthly_rf, axis=0)

    print(f"✓ Excess Returns calculated")
    print(f"  Total months aligned: {len(excess_returns)}")
    if len(missing_months) > 0:
        print(f"  (of which {len(missing_months)} use forward-filled FF factors)")

    return excess_returns, monthly_rf, monthly_factors

# =============================================================================
# STEP 3B: CALCULATE MOMENTUM (12-1)
# =============================================================================

def calculate_momentum(monthly_returns, lookback=12):

        print(f"\n{'=' * 60}")
        print(f"📈 CALCULATING MOMENTUM (12-1)")
        print(f"{'=' * 60}")

        # DEBUG: Check input
        print(f"  Input shape: {monthly_returns.shape}")
        print(f"  Input has NaN: {monthly_returns.isna().sum().sum()}")

        # Calculate cumulative return of last 12 months
        # Exclude last month (t-1 to t-12, not t to t-11)
        momentum = monthly_returns.shift(1).rolling(window=lookback - 1).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )

        print(f"✓ Momentum calculated: {momentum.shape}")
        print(f"  First {lookback} months are NaN (rolling window)")
        print(f"  Valid values: {momentum.notna().sum().sum()}")

        return momentum

# =============================================================================
# STEP 4: ALPHA FILTER
# =============================================================================

def get_alpha_filter(excess_returns, monthly_factors, window=36):
    import statsmodels.api as sm

    print(f"\n{'=' * 60}")
    print(f"🔍 RUNNING ROLLING ALPHA FILTER (FF5F)")
    print(f"{'=' * 60}")

    alpha_passed = {}
    total_months = len(excess_returns)

    for i in range(window, total_months):
        current_date = excess_returns.index[i]
        eligible_tickers = []
        monthly_alphas = {}

        y_window = excess_returns.iloc[i - window:i]
        x_window = monthly_factors.iloc[i - window:i]
        x_window_with_const = sm.add_constant(x_window)

        for ticker in excess_returns.columns:
            try:
                valid_data = y_window[ticker].dropna()
                if len(valid_data) < window: continue

                model = sm.OLS(valid_data, x_window_with_const.loc[valid_data.index]).fit()
                alpha = model.params['const']

                if alpha > 0:
                    monthly_alphas[ticker] = alpha
            except:
                continue

        if monthly_alphas:
            sorted_alphas = sorted(monthly_alphas.items(), key=lambda x: x[1], reverse=True)
            eligible_tickers = [ticker for ticker, alpha in sorted_alphas]
        else:
            eligible_tickers = []

        alpha_passed[current_date] = eligible_tickers

        if (i - window + 1) % 12 == 0 or i == total_months - 1:
            print(f"  📅 Processed: {current_date.date()} | Unique Alpha+: {len(eligible_tickers)}")

    print(f"✓ Alpha filter complete for {len(alpha_passed)} months.")
    return alpha_passed

# =============================================================================
# STEP 5: CALCULATE WEIGHTS
# =============================================================================

def apply_sector_filters(active_tickers, sector_mapping, sector_limits, min_weight=0.05):

    # Group tickers by sector
    sector_groups = {}
    unknown_tickers = []

    for ticker in active_tickers:
        sector = sector_mapping.get(ticker, 'Unknown')
        if sector != 'Unknown':
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(ticker)
        else:
            unknown_tickers.append(ticker)

    # DEBUG: show what was found
    if len(unknown_tickers) > 0:
        print(f"  ⚠️  Unknown sectors for: {unknown_tickers[:5]}...")  # primi 5

    # Select tickers for each sector respecting limits
    final_tickers = []
    selected_by_sector = {}

    for sector, sector_tickers in sector_groups.items():
        # Get sector limit, or 10% default
        max_sector_weight = sector_limits.get(sector, 0.10)

        # Maximum number of tickers = max_sector_weight / min_weight
        max_tickers_in_sector = max(1, int(max_sector_weight / min_weight))

        # Take the top N tickers for the sector
        selected = sector_tickers[:max_tickers_in_sector]
        selected_by_sector[sector] = selected
        final_tickers.extend(selected)

    return final_tickers, selected_by_sector

def calculate_weights(dict_alpha_plus, prices_daily, momentum, sector_mapping,
                      sector_limits, min_weight=0.05):

    print(f"\n{'=' * 60}")
    print(f"⚖️ CALCULATING PORTFOLIO WEIGHTS WITH SECTOR FILTERS")
    print(f"{'=' * 60}")
    print(f"  Minimum weight per ticker: {min_weight * 100:.0f}%")

    portfolio_weights = {}
    monthly_prices = prices_daily.resample('ME').last()
    monthly_returns = monthly_prices.pct_change()
    volatility = monthly_returns.rolling(window=12).std()

    valid_months_count = 0

    for date, active_tickers in dict_alpha_plus.items():

        if not active_tickers:
            continue

        # Apply sector filters
        filtered_tickers, sector_breakdown = apply_sector_filters(
            active_tickers, sector_mapping, sector_limits, min_weight
        )

        if not filtered_tickers:
            continue

        # Verify tickers are in the dataset
        available_tickers = [t for t in filtered_tickers if t in momentum.columns]

        if not available_tickers:
            continue

        # Calculate weights based on momentum and volatility
        try:
            mom_values = momentum.loc[date, available_tickers]
            vol_values = volatility.loc[date, available_tickers]
        except KeyError:
            # Date not available
            continue

        # Remove NaNs
        valid_mask = mom_values.notna() & vol_values.notna() & (vol_values > 0)
        mom_values = mom_values[valid_mask]
        vol_values = vol_values[valid_mask]

        if len(mom_values) == 0:
            continue

        # Score: momentum / volatility
        score = mom_values / vol_values
        score = score.clip(lower=0)

        if score.sum() <= 0:
            continue

        # Normalize to 1
        w_normalized = score / score.sum()

        # If at least one valid ticker exists
        if len(w_normalized) > 0:
            # Take top N tickers based on minimum weight
            max_positions = int(1.0 / min_weight)

            # If more tickers than max, take the best ones
            if len(w_normalized) > max_positions:
                w_normalized = w_normalized.nlargest(max_positions)
                w_normalized = w_normalized / w_normalized.sum()

            # Apply minimum weight filter
            w_filtered = w_normalized[w_normalized >= min_weight * 0.8]

            if len(w_filtered) == 0:
                # If all below minimum, take top 10 regardless
                w_filtered = w_normalized.nlargest(min(10, len(w_normalized)))

            # Re-normalize
            w_final = w_filtered / w_filtered.sum()

            # Round to clean integers
            w_integers = round_weights_integers(w_final, 100)
            portfolio_weights[date] = w_integers / 100
            valid_months_count += 1

    print(f"✓ Weights calculated for {len(portfolio_weights)} months.")
    print(f"✓ Valid portfolio months: {valid_months_count}/{len(dict_alpha_plus)}")

    if valid_months_count < 10:
        print(f"⚠️  WARNING: Very few valid months! Check data alignment.")

    return portfolio_weights

def round_weights_integers(weights_series, target_total):
    # Convert to integers
    rounded = (weights_series * target_total).round().astype(int)

    # Correction: if sum is 99 or 101 due to rounding,
    # add/subtract the difference from the asset with the highest weight
    diff = target_total - rounded.sum()
    if diff != 0:
        rounded[rounded.idxmax()] += diff

    return rounded


# =============================================================================
# SILENT WEIGHT CALCULATION (for Monte Carlo simulations)
# =============================================================================

def calculate_weights_silent(dict_alpha_plus, prices_daily, momentum, sector_mapping,
                             sector_limits, min_weight=0.05):
    """
    Same logic as calculate_weights but without print statements.
    Used by Monte Carlo robustness simulations to avoid console spam.
    
    Args:
        dict_alpha_plus: Dictionary of alpha-filtered tickers per date
        prices_daily: DataFrame of daily prices
        momentum: DataFrame of momentum scores
        sector_mapping: Dictionary mapping ticker -> sector
        sector_limits: Dictionary of sector allocation limits
        min_weight: Minimum weight per stock
    
    Returns:
        portfolio_weights: Dictionary with dates as keys, weight Series as values
    """
    portfolio_weights = {}
    monthly_prices = prices_daily.resample('ME').last()
    monthly_returns = monthly_prices.pct_change()
    volatility = monthly_returns.rolling(window=12).std()

    for date, active_tickers in dict_alpha_plus.items():

        if not active_tickers:
            continue

        # Apply sector filters
        filtered_tickers, sector_breakdown = apply_sector_filters(
            active_tickers, sector_mapping, sector_limits, min_weight
        )

        if not filtered_tickers:
            continue

        # Verify tickers are in the dataset
        available_tickers = [t for t in filtered_tickers if t in momentum.columns]

        if not available_tickers:
            continue

        # Calculate weights based on momentum and volatility
        try:
            mom_values = momentum.loc[date, available_tickers]
            vol_values = volatility.loc[date, available_tickers]
        except KeyError:
            continue

        # Remove NaNs
        valid_mask = mom_values.notna() & vol_values.notna() & (vol_values > 0)
        mom_values = mom_values[valid_mask]
        vol_values = vol_values[valid_mask]

        if len(mom_values) == 0:
            continue

        # Score: momentum / volatility
        score = mom_values / vol_values
        score = score.clip(lower=0)

        if score.sum() <= 0:
            continue

        # Normalize to 1
        w_normalized = score / score.sum()

        if len(w_normalized) > 0:
            max_positions = int(1.0 / min_weight)

            if len(w_normalized) > max_positions:
                w_normalized = w_normalized.nlargest(max_positions)
                w_normalized = w_normalized / w_normalized.sum()

            w_filtered = w_normalized[w_normalized >= min_weight * 0.8]

            if len(w_filtered) == 0:
                w_filtered = w_normalized.nlargest(min(10, len(w_normalized)))

            w_final = w_filtered / w_filtered.sum()

            w_integers = round_weights_integers(w_final, 100)
            portfolio_weights[date] = w_integers / 100

    return portfolio_weights

# =============================================================================
# STEP 6: BACKTEST
# =============================================================================

def run_backtest(portfolio_weights, monthly_returns):

    print(f"\n{'=' * 60}")
    print(f"📈 RUNNING STRATEGY BACKTEST")
    print(f"{'=' * 60}")

    # List to store monthly portfolio returns
    strategy_returns = pd.Series(index=monthly_returns.index, dtype=float)

    # Iterate over months with defined weights
    for date in sorted(portfolio_weights.keys()):
        next_month_idx = monthly_returns.index.get_indexer([date], method='bfill')[0] + 1

        if next_month_idx < len(monthly_returns):
            target_date = monthly_returns.index[next_month_idx]
            weights = portfolio_weights[date]
            rets = monthly_returns.loc[target_date, weights.index]

            # Portfolio return for that month: sum(weight * return)
            strategy_returns[target_date] = (weights * rets).sum()

    # Cleaning and Equity Curve calculation (starting from 100)
    strategy_returns = strategy_returns.dropna()
    equity_curve = (1 + strategy_returns).cumprod() * 100

    # Basic Metrics Calculation
    total_return = (equity_curve.iloc[-1] / 100) - 1
    ann_return = (1 + total_return) ** (12 / len(strategy_returns)) - 1
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * (12 ** 0.5)

    print(f"✓ Backtest done:")
    print(f"  - Total Return: {total_return:.2%}")
    print(f"  - Annual Return: {ann_return:.2%}")
    print(f"  - Sharpe Ratio: {sharpe:.2f}")

    return equity_curve, strategy_returns

# =============================================================================
# STEP 6: TELEGRAM
# =============================================================================

def send_telegram_alert(token, chat_id, date, weights):
    message = f"🚀 *INVESTING STRATEGY: NEW REBALANCE*\n"
    message += f"📅 Date: {date.date()}\n\n"
    message += f"📊 *Target Portfolio Holdings:*\n"

    weights_sorted = weights.sort_values(ascending=False)
    for asset, weight in weights_sorted.items():
        if weight > 0:
            message += f"• {asset}: {weight * 100:.2f}%\n"

    message += f"\n🔄 _Please update your brokerage account positions._"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, json=payload)
        print("✅ Alert sent to Telegram!")
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # CONFIGURATION
    TICKERS = []
    SECTOR_MAPPING = {}

    try:
        print("\n📥 Downloading S&P 500 tickers from Wikipedia...")

        # Download Wikipedia table using User-Agent
        URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        tables = pd.read_html(URL, storage_options=headers)
        df_tickers = tables[0]

        # Create ticker -> sector mapping dictionary
        SECTOR_MAPPING = dict(zip(
            df_tickers['Symbol'].str.replace('.', '-', regex=False),
            df_tickers['GICS Sector']
        ))

        # Extract tickers
        TICKERS = df_tickers['Symbol'].str.replace('.', '-', regex=False).tolist()

        print(f"✓ Successfully downloaded {len(TICKERS)} tickers from S&P 500")
        print(f"✓ Sectors available: {df_tickers['GICS Sector'].unique().tolist()}")

    except Exception as e:
        print(f"❌ Error downloading tickers from Wikipedia: {e}")

    START_DATE = '2015-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # To be rebalanced each time based on the period
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

    MIN_WEIGHT = 0.05  # Minimum weight 5% per ticker

    print("\n" + "=" * 70)
    print(f"🚀 BACKTEST S&P 500 ({len(TICKERS)} ASSETS)")
    print("=" * 70)

    # 1-3: DOWNLOAD DATA
    PRICES_DAILY, failed = download_daily_prices(TICKERS, START_DATE, END_DATE)

    FF_FACTORS = get_fama_french_factors(START_DATE, END_DATE)

    EXCESS_RETURNS, MONTHLY_RF, MONTHLY_FACTORS = create_monthly_returns(
        PRICES_DAILY, FF_FACTORS
    )

    monthly_prices_for_momentum = PRICES_DAILY.resample('ME').last()
    monthly_returns_for_momentum = monthly_prices_for_momentum.pct_change()
    MOMENTUM = calculate_momentum(monthly_returns_for_momentum, lookback=12)

    # 4-5: STRATEGY (FILTERS AND WEIGHTS)
    DICT_ALPHA_PLUS = get_alpha_filter(EXCESS_RETURNS, MONTHLY_FACTORS, window=36)

    PORTFOLIO_WEIGHTS = calculate_weights(
        DICT_ALPHA_PLUS,
        PRICES_DAILY,
        MOMENTUM,
        SECTOR_MAPPING,
        SECTOR_LIMITS,
        MIN_WEIGHT
    )

    # 6-7: BACKTEST E BENCHMARK
    MONTHLY_PRICES = PRICES_DAILY.resample('ME').last()
    MONTHLY_RETS_REAL = MONTHLY_PRICES.pct_change()

    equity_curve, strategy_returns = run_backtest(PORTFOLIO_WEIGHTS, MONTHLY_RETS_REAL)

    # Benchmark SPY
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)['Close']
    spy_monthly_rets = spy.resample('ME').last().pct_change()

    common_dates = equity_curve.index
    spy_equity = (1 + spy_monthly_rets.loc[common_dates]).cumprod() * 100

    # 8: FINAL REPORT AND PLOT
    print(f"\n{'=' * 70}")
    print(f"🏆 PERFORMANCE REPORT S&P 500 UNIVERSE")
    print(f"{'=' * 70}")

    final_strat = float(equity_curve.iloc[-1])

    try:
        final_spy = float(spy_equity.iloc[-1].item())
    except:
        final_spy = float(spy_equity.iloc[-1])

    print(f"✓ Strategy Final Equity: {final_strat:.2f}")
    print(f"✓ S&P 500 Final Equity:   {final_spy:.2f}")

    strat_ann = (final_strat / 100) ** (12 / len(equity_curve)) - 1
    spy_ann = (final_spy / 100) ** (12 / len(spy_equity)) - 1
    print(f"✓ Annual Alpha Generated: {(strat_ann - spy_ann):.2%}")

    plt.figure(figsize=(12, 7))
    plt.plot(equity_curve, label=f'Strategy (Ann: {strat_ann:.1%})', color='#1f77b4', lw=2)
    plt.plot(spy_equity, label=f'S&P 500 - SPY (Ann: {spy_ann:.1%})', color='#7f7f7f', linestyle='--', alpha=0.7)

    plt.title('Backtest: FF5F Alpha + Momentum + Low Vol', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value (Base 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 9: EXPORT RESULTS TO EXCEL
    try:
        with pd.ExcelWriter('strategy_results.xlsx') as writer:
            equity_df = equity_curve.to_frame('Strategy_Value')
            equity_df['Benchmark_Value'] = spy_equity
            equity_df.to_excel(writer, sheet_name='Performance')

            selected_assets_data = []
            for date, assets in DICT_ALPHA_PLUS.items():
                selected_assets_data.append({'Date': date, 'Assets': ", ".join(assets)})

            pd.DataFrame(selected_assets_data).to_excel(writer, sheet_name='Selected_Assets', index=False)
        print("\n✅ Results exported to 'strategy_results.xlsx'")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")

# =============================================================================
# GET S&P 500 TICKERS
# =============================================================================

def get_sp500_tickers():
    """Download S&P 500 tickers from Wikipedia"""
    try:
        print("\n📥 Downloading S&P 500 tickers from Wikipedia...")

        URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        tables = pd.read_html(URL, storage_options=headers)
        df_tickers = tables[0]

        # Create ticker -> sector mapping dictionary
        sector_mapping = dict(zip(
            df_tickers['Symbol'].str.replace('.', '-', regex=False),
            df_tickers['GICS Sector']
        ))

        # Extract tickers
        tickers = df_tickers['Symbol'].str.replace('.', '-', regex=False).tolist()

        print(f"✓ Successfully downloaded {len(tickers)} tickers from S&P 500")
        print(f"✓ Sectors available: {df_tickers['GICS Sector'].unique().tolist()}")

        return tickers, sector_mapping

    except Exception as e:
        print(f"❌ Error downloading tickers from Wikipedia: {e}")
        return [], {}

# =============================================================================
# RUN STOCK STRATEGY
# =============================================================================

def run_stock_strategy(start_date, end_date, sector_limits, min_weight=0.05, alpha_window=36, 
                      momentum_lookback=12, round_to_integer=True, return_intermediate_data=False):
    """
    Complete stock selection strategy with ORIGINAL WINNING LOGIC
    
    Args:
        start_date: Start date
        end_date: End date
        sector_limits: Dictionary of sector limits
        min_weight: Minimum weight per stock
        alpha_window: Rolling window for alpha calculation
        momentum_lookback: Momentum lookback period
        round_to_integer: Round weights to integer percentages (not used, always True)
        return_intermediate_data: If True, also return dict_alpha_plus, prices_daily,
                                  momentum, sector_mapping for robustness MC
    
    Returns: 
        portfolio_weights, monthly_returns
        (if return_intermediate_data=True, also returns intermediate_data dict)
    """
    print("\n" + "=" * 70)
    print(f"🚀 RUNNING STOCK STRATEGY")
    print("=" * 70)

    # Get S&P 500 tickers
    tickers, sector_mapping = get_sp500_tickers()

    if not tickers:
        print("❌ Failed to get tickers")
        if return_intermediate_data:
            return {}, pd.DataFrame(), None
        return {}, pd.DataFrame()

    # Download data
    prices_daily, failed = download_daily_prices(tickers, start_date, end_date)
    ff_factors = get_fama_french_factors(start_date, end_date)

    if ff_factors is None:
        print("❌ Failed to download Fama-French factors")
        if return_intermediate_data:
            return {}, pd.DataFrame(), None
        return {}, pd.DataFrame()

    # Calculate returns
    excess_returns, monthly_rf, monthly_factors = create_monthly_returns(prices_daily, ff_factors)

    # Calculate momentum
    monthly_prices_for_momentum = prices_daily.resample('ME').last()
    monthly_returns_for_momentum = monthly_prices_for_momentum.pct_change()
    momentum = calculate_momentum(monthly_returns_for_momentum, lookback=momentum_lookback)

    # Apply filters and calculate weights
    dict_alpha_plus = get_alpha_filter(excess_returns, monthly_factors, window=alpha_window)

    portfolio_weights = calculate_weights(
        dict_alpha_plus,
        prices_daily,
        momentum,
        sector_mapping,
        sector_limits,
        min_weight
    )

    # Calculate monthly returns for backtesting
    monthly_prices = prices_daily.resample('ME').last()
    monthly_returns = monthly_prices.pct_change()

    print(f"\n✓ Stock strategy complete!")
    print(f"  - Portfolio rebalancing dates: {len(portfolio_weights)}")

    if return_intermediate_data:
        intermediate_data = {
            'dict_alpha_plus': dict_alpha_plus,
            'prices_daily': prices_daily,
            'momentum': momentum,
            'sector_mapping': sector_mapping,
        }
        return portfolio_weights, monthly_returns, intermediate_data

    return portfolio_weights, monthly_returns


if __name__ == "__main__":

    # CONFIGURATION
    TICKERS = []
    SECTOR_MAPPING = {}

    try:
        print("\n📥 Downloading S&P 500 tickers from Wikipedia...")

        # Download Wikipedia table using User-Agent
        URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        tables = pd.read_html(URL, storage_options=headers)
        df_tickers = tables[0]

        # Create ticker -> sector mapping dictionary
        SECTOR_MAPPING = dict(zip(
            df_tickers['Symbol'].str.replace('.', '-', regex=False),
            df_tickers['GICS Sector']
        ))

        # Extract tickers
        TICKERS = df_tickers['Symbol'].str.replace('.', '-', regex=False).tolist()

        print(f"✓ Successfully downloaded {len(TICKERS)} tickers from S&P 500")
        print(f"✓ Sectors available: {df_tickers['GICS Sector'].unique().tolist()}")

    except Exception as e:
        print(f"❌ Error downloading tickers from Wikipedia: {e}")

    START_DATE = '2015-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # To be rebalanced each time based on the period
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

    MIN_WEIGHT = 0.05  # Minimum weight 5% per ticker

    print("\n" + "=" * 70)
    print(f"🚀 BACKTEST S&P 500 ({len(TICKERS)} ASSETS)")
    print("=" * 70)

    # 1-3: DOWNLOAD DATA
    PRICES_DAILY, failed = download_daily_prices(TICKERS, START_DATE, END_DATE)

    FF_FACTORS = get_fama_french_factors(START_DATE, END_DATE)

    EXCESS_RETURNS, MONTHLY_RF, MONTHLY_FACTORS = create_monthly_returns(
        PRICES_DAILY, FF_FACTORS
    )

    monthly_prices_for_momentum = PRICES_DAILY.resample('ME').last()
    monthly_returns_for_momentum = monthly_prices_for_momentum.pct_change()
    MOMENTUM = calculate_momentum(monthly_returns_for_momentum, lookback=12)

    # 4-5: STRATEGY (FILTERS AND WEIGHTS)
    DICT_ALPHA_PLUS = get_alpha_filter(EXCESS_RETURNS, MONTHLY_FACTORS, window=36)

    PORTFOLIO_WEIGHTS = calculate_weights(
        DICT_ALPHA_PLUS,
        PRICES_DAILY,
        MOMENTUM,
        SECTOR_MAPPING,
        SECTOR_LIMITS,
        MIN_WEIGHT
    )

    # 6-7: BACKTEST E BENCHMARK
    MONTHLY_PRICES = PRICES_DAILY.resample('ME').last()
    MONTHLY_RETS_REAL = MONTHLY_PRICES.pct_change()

    equity_curve, strategy_returns = run_backtest(PORTFOLIO_WEIGHTS, MONTHLY_RETS_REAL)

    # Benchmark SPY
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)['Close']
    spy_monthly_rets = spy.resample('ME').last().pct_change()

    common_dates = equity_curve.index
    spy_equity = (1 + spy_monthly_rets.loc[common_dates]).cumprod() * 100

    # 8: FINAL REPORT AND PLOT
    print(f"\n{'=' * 70}")
    print(f"🏆 PERFORMANCE REPORT S&P 500 UNIVERSE")
    print(f"{'=' * 70}")

    final_strat = float(equity_curve.iloc[-1])

    try:
        final_spy = float(spy_equity.iloc[-1].item())
    except:
        final_spy = float(spy_equity.iloc[-1])

    print(f"✓ Strategy Final Equity: {final_strat:.2f}")
    print(f"✓ S&P 500 Final Equity:   {final_spy:.2f}")

    strat_ann = (final_strat / 100) ** (12 / len(equity_curve)) - 1
    spy_ann = (final_spy / 100) ** (12 / len(spy_equity)) - 1
    print(f"✓ Annual Alpha Generated: {(strat_ann - spy_ann):.2%}")

    plt.figure(figsize=(12, 7))
    plt.plot(equity_curve, label=f'Strategy (Ann: {strat_ann:.1%})', color='#1f77b4', lw=2)
    plt.plot(spy_equity, label=f'S&P 500 - SPY (Ann: {spy_ann:.1%})', color='#7f7f7f', linestyle='--', alpha=0.7)

    plt.title('Backtest: FF5F Alpha + Momentum + Low Vol', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value (Base 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 9: EXPORT RESULTS TO EXCEL
    try:
        with pd.ExcelWriter('strategy_results.xlsx') as writer:
            equity_df = equity_curve.to_frame('Strategy_Value')
            equity_df['Benchmark_Value'] = spy_equity
            equity_df.to_excel(writer, sheet_name='Performance')

            selected_assets_data = []
            for date, assets in DICT_ALPHA_PLUS.items():
                selected_assets_data.append({'Date': date, 'Assets': ", ".join(assets)})

            pd.DataFrame(selected_assets_data).to_excel(writer, sheet_name='Selected_Assets', index=False)
        print("\n✅ Results exported to 'strategy_results.xlsx'")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")

    # 10: ALERT ON TELEGRAM
    available_dates = sorted(PORTFOLIO_WEIGHTS.keys())
    last_available_date = available_dates[-1]
    last_weights = PORTFOLIO_WEIGHTS[last_available_date]

    current_display_date = datetime.now()

    MY_TOKEN = "8514820447:AAH927K_PHktau1fYlnTbnNpGO6OBCA4gDE"
    MY_CHAT_ID = "217484630"

    send_telegram_alert(MY_TOKEN, MY_CHAT_ID, current_display_date, last_weights)