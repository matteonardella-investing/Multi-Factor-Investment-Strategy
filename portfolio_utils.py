import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_backtest(portfolio_weights, monthly_returns, strategy_name="Strategy"):
    """
    Run backtest for a given portfolio
    
    Args:
        portfolio_weights: Dictionary with dates as keys and weight Series as values
        monthly_returns: DataFrame with monthly returns
        strategy_name: Name for display
    
    Returns:
        equity_curve: Series with portfolio value over time
        strategy_returns: Series with monthly portfolio returns
    """
    print(f"\n{'=' * 60}")
    print(f"📈 RUNNING {strategy_name.upper()} BACKTEST")
    print(f"{'=' * 60}")

    # List to store monthly portfolio returns
    strategy_returns = pd.Series(index=monthly_returns.index, dtype=float)

    # Iterate over months with defined weights
    for date in sorted(portfolio_weights.keys()):
        next_month_idx = monthly_returns.index.get_indexer([date], method='bfill')[0] + 1

        if next_month_idx < len(monthly_returns):
            target_date = monthly_returns.index[next_month_idx]
            weights = portfolio_weights[date]
            
            # Get returns for the assets in the portfolio
            available_assets = weights.index.intersection(monthly_returns.columns)
            if len(available_assets) == 0:
                continue
                
            weights_available = weights[available_assets]
            rets = monthly_returns.loc[target_date, available_assets]

            # Portfolio return for that month: sum(weight * return)
            strategy_returns[target_date] = (weights_available * rets).sum()

    # Cleaning and Equity Curve calculation (starting from 100)
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        print(f"❌ No valid returns for {strategy_name}")
        return pd.Series(), pd.Series()
    
    equity_curve = (1 + strategy_returns).cumprod() * 100

    # Basic Metrics Calculation
    total_return = (equity_curve.iloc[-1] / 100) - 1
    ann_return = (1 + total_return) ** (12 / len(strategy_returns)) - 1
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * (12 ** 0.5) if strategy_returns.std() > 0 else 0

    print(f"✓ Backtest done:")
    print(f"  - Total Return: {total_return:.2%}")
    print(f"  - Annual Return: {ann_return:.2%}")
    print(f"  - Sharpe Ratio: {sharpe:.2f}")

    return equity_curve, strategy_returns


def get_benchmark(ticker, start_date, end_date):
    """
    Download and calculate benchmark returns
    
    Args:
        ticker: Benchmark ticker (e.g., 'SPY')
        start_date: Start date
        end_date: End date
    
    Returns:
        monthly_returns: Series of monthly benchmark returns
    """
    print(f"\n📊 Downloading benchmark: {ticker}")
    
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        monthly_rets = data.resample('ME').last().pct_change()
        print(f"✓ Benchmark downloaded successfully")
        return monthly_rets
    except Exception as e:
        print(f"❌ Error downloading benchmark: {e}")
        return pd.Series()

# =============================================================================
# EXCEL EXPORT
# =============================================================================

def export_to_excel(filename, equity_curve, portfolio_weights, benchmark=None):
    """
    Export results to Excel file
    
    Args:
        filename: Output filename
        equity_curve: Portfolio equity curve Series
        portfolio_weights: Dictionary of portfolio weights
        benchmark: Optional benchmark Series
    """
    print(f"\n{'=' * 60}")
    print(f"💾 EXPORTING RESULTS TO EXCEL")
    print(f"{'=' * 60}")

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Sheet 1: Performance comparison
            performance_df = pd.DataFrame()
            performance_df['Portfolio'] = equity_curve
            
            if benchmark is not None and not benchmark.empty:
                # Calculate benchmark equity curve
                benchmark_clean = benchmark.dropna()
                if len(benchmark_clean) > 0:
                    benchmark_equity = (1 + benchmark_clean).cumprod() * 100
                    # Align with performance dates
                    common_dates = performance_df.index.intersection(benchmark_equity.index)
                    if len(common_dates) > 0:
                        performance_df['Benchmark'] = benchmark_equity.loc[common_dates]
            
            performance_df.to_excel(writer, sheet_name='Performance')
            
            # Sheet 2: Latest Portfolio Weights
            if portfolio_weights:
                latest_date = max(portfolio_weights.keys())
                latest_weights = portfolio_weights[latest_date].sort_values(ascending=False)
                
                weights_df = pd.DataFrame({
                    'Asset': latest_weights.index,
                    'Weight': latest_weights.values,
                    'Weight_Pct': (latest_weights.values * 100).round(2)
                })
                weights_df.to_excel(writer, sheet_name='Latest_Weights', index=False)
                
            # Sheet 3: All Rebalancing Dates and Weights
            all_rebalances = []
            for date, weights in sorted(portfolio_weights.items()):
                for asset, weight in weights.items():
                    all_rebalances.append({
                        'Date': date,
                        'Asset': asset,
                        'Weight': weight,
                        'Weight_Pct': weight * 100
                    })
            
            if all_rebalances:
                rebalances_df = pd.DataFrame(all_rebalances)
                rebalances_df.to_excel(writer, sheet_name='All_Rebalances', index=False)

        print(f"✅ Results exported to '{filename}'")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")


# =============================================================================
# TELEGRAM ALERTS
# =============================================================================

def send_telegram_alert(token, chat_id, portfolio_weights):
    """
    Send portfolio rebalancing alert to Telegram
    
    Args:
        token: Telegram bot token
        chat_id: Telegram chat ID
        portfolio_weights: Dictionary with portfolio weights
    """
    if not portfolio_weights:
        print("⚠️ No portfolio weights to send")
        return

    # Get latest portfolio
    latest_date = max(portfolio_weights.keys())
    latest_weights = portfolio_weights[latest_date].sort_values(ascending=False)

    # Build message
    message = f"🚀 *PORTFOLIO REBALANCING ALERT*\n"
    message += f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
    
    message += f"📈 *STOCKS ({len(latest_weights)} positions):*\n"
    for asset, weight in latest_weights.items():
        message += f"• {asset}: {weight * 100:.0f}%\n"
    
    message += f"\n🔄 _Please update your brokerage account positions._"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Alert sent to Telegram!")
        else:
            print(f"⚠️ Telegram response: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_performance(equity_curve, benchmark=None, title="Portfolio Performance"):
    """
    Plot performance comparison
    
    Args:
        equity_curve: Portfolio equity curve Series
        benchmark: Optional benchmark Series
        title: Plot title
    """
    print(f"\n📊 Generating performance plot...")

    plt.figure(figsize=(14, 8))

    # Plot portfolio
    final_value = equity_curve.iloc[-1]
    if hasattr(final_value, 'item'):
        final_value = final_value.item()
    else:
        final_value = float(final_value)
        
    total_return = (final_value / 100) - 1
    ann_return = (1 + total_return) ** (12 / len(equity_curve)) - 1
    
    plt.plot(equity_curve.index, equity_curve.values, 
             label=f'Portfolio (Ann: {ann_return:.1%})', lw=2.5, color='#2E86AB')

    # Plot benchmark
    if benchmark is not None and not benchmark.empty:
        benchmark_clean = benchmark.dropna()
        if len(benchmark_clean) > 0:
            # Align benchmark returns with portfolio dates
            benchmark_returns_aligned = benchmark_clean.reindex(equity_curve.index).dropna()
            
            if len(benchmark_returns_aligned) > 0:
                # Recalculate equity curve starting from 100
                benchmark_equity_aligned = (1 + benchmark_returns_aligned).cumprod() * 100
                
                # Ensure scalar value
                bm_final_value = benchmark_equity_aligned.iloc[-1]
                if hasattr(bm_final_value, 'item'):
                    bm_final_value = bm_final_value.item()
                else:
                    bm_final_value = float(bm_final_value)
                
                # Calculate return from base 100
                total_return_bm = (bm_final_value / 100) - 1
                years = len(benchmark_returns_aligned) / 12
                ann_return_bm = (1 + total_return_bm) ** (1 / years) - 1
                
                plt.plot(benchmark_equity_aligned.index, benchmark_equity_aligned.values, 
                        label=f'Benchmark (Ann: {ann_return_bm:.1%})', 
                        linestyle='--', alpha=0.7, lw=2, color='#A23B72')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (Base 100)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'portfolio_performance.png'")
    
    plt.show()
