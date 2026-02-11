import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# =============================================================================
# TRANSACTION COST FUNCTIONS
# =============================================================================

def calculate_transaction_costs(trades, portfolio_value, bid_ask_spread, 
                                commission_per_trade, commission_per_share, 
                                market_impact_factor):
    """
    Calculate transaction costs for a set of trades
    
    Args:
        trades: Series of position changes (new_weight - old_weight)
        portfolio_value: Total portfolio value in dollars
        bid_ask_spread: Bid-ask spread as decimal (e.g., 0.0005 = 0.05%)
        commission_per_trade: Fixed commission per trade in dollars
        commission_per_share: Commission per share in dollars
        market_impact_factor: Market impact coefficient
    
    Returns:
        total_cost: Total transaction cost as percentage of portfolio
        cost_breakdown: Dictionary with cost components
    """
    # Only consider actual trades (non-zero changes)
    actual_trades = trades[trades.abs() > 0]
    
    if len(actual_trades) == 0:
        return 0.0, {'bid_ask': 0, 'commission_fixed': 0, 'commission_shares': 0, 'market_impact': 0}
    
    # 1. Bid-Ask Spread Cost
    # Applied to the dollar value traded
    bid_ask_cost = (actual_trades.abs() * bid_ask_spread).sum()
    
    # 2. Fixed Commission Cost
    num_trades = len(actual_trades)
    commission_fixed_cost = (num_trades * commission_per_trade) / portfolio_value
    
    # 3. Per-Share Commission Cost
    # Estimate number of shares traded (assuming average stock price of $100)
    avg_stock_price = 100  # Reasonable estimate for S&P 500 stocks
    total_shares_traded = (actual_trades.abs() * portfolio_value / avg_stock_price).sum()
    commission_shares_cost = (total_shares_traded * commission_per_share) / portfolio_value
    
    # 4. Market Impact Cost
    # Scales with trade size: larger trades have proportionally higher impact
    # Cost per trade = market_impact_factor * (trade_size_pct)^1.5
    market_impact_cost = (market_impact_factor * (actual_trades.abs() ** 1.5)).sum()
    
    # Total cost as percentage
    total_cost = bid_ask_cost + commission_fixed_cost + commission_shares_cost + market_impact_cost
    
    cost_breakdown = {
        'bid_ask': bid_ask_cost,
        'commission_fixed': commission_fixed_cost,
        'commission_shares': commission_shares_cost,
        'market_impact': market_impact_cost,
        'num_trades': num_trades
    }
    
    return total_cost, cost_breakdown


def calculate_turnover(old_weights, new_weights):
    """
    Calculate portfolio turnover
    
    Turnover = sum(|new_weight - old_weight|) / 2
    
    Args:
        old_weights: Series of previous weights
        new_weights: Series of new target weights
    
    Returns:
        turnover: Portfolio turnover as decimal
    """
    # Align indices (handle additions/removals)
    all_assets = old_weights.index.union(new_weights.index)
    
    old_aligned = old_weights.reindex(all_assets, fill_value=0)
    new_aligned = new_weights.reindex(all_assets, fill_value=0)
    
    # Calculate turnover
    turnover = (new_aligned - old_aligned).abs().sum() / 2
    
    return turnover


def apply_turnover_constraint(old_weights, new_weights, max_turnover, min_trade_threshold=0.01, min_position_weight=None):
    """
    Apply turnover constraint by scaling down trades if necessary
    PRIORITY: Turnover limit is more important than minimum position weight
    
    Args:
        old_weights: Series of previous weights
        new_weights: Series of new target weights
        max_turnover: Maximum allowed turnover
        min_trade_threshold: Minimum weight change to execute trade
        min_position_weight: Minimum weight per position (advisory only)
    
    Returns:
        constrained_weights: Series of constrained weights
        actual_turnover: Actual turnover after constraint
        was_constrained: Boolean indicating if constraint was applied
    """
    # Align indices
    all_assets = old_weights.index.union(new_weights.index)
    old_aligned = old_weights.reindex(all_assets, fill_value=0)
    new_aligned = new_weights.reindex(all_assets, fill_value=0)
    
    # Calculate target turnover
    target_turnover = calculate_turnover(old_weights, new_weights)
    
    # Determine if we need to constrain
    was_constrained = target_turnover > max_turnover
    
    if was_constrained:
        # Scale down ALL trades proportionally to meet turnover constraint
        scale_factor = max_turnover / target_turnover
        trades = new_aligned - old_aligned
        scaled_trades = trades * scale_factor
        constrained_weights = old_aligned + scaled_trades
    else:
        # No constraint needed, but filter small trades
        trade_sizes = (new_aligned - old_aligned).abs()
        large_trades_mask = trade_sizes >= min_trade_threshold
        
        constrained_weights = old_aligned.copy()
        constrained_weights[large_trades_mask] = new_aligned[large_trades_mask]
    
    # Clean up: remove negatives and very small positions
    constrained_weights = constrained_weights.clip(lower=0)
    constrained_weights = constrained_weights[constrained_weights > 1e-6]
    
    # Final normalization to ensure sum = 1
    if constrained_weights.sum() > 0:
        constrained_weights = constrained_weights / constrained_weights.sum()
    
    # Calculate actual turnover achieved
    actual_turnover = calculate_turnover(old_weights, constrained_weights)
    
    # Verify we haven't exceeded the limit (should never happen)
    if actual_turnover > max_turnover * 1.01:  # Allow 1% tolerance
        print(f"⚠️ WARNING: Turnover {actual_turnover:.2%} exceeds limit {max_turnover:.2%}!")
    
    return constrained_weights, actual_turnover, was_constrained


# =============================================================================
# BACKTEST FUNCTIONS (UPDATED WITH TRANSACTION COSTS)
# =============================================================================

def run_backtest(portfolio_weights, monthly_returns, strategy_name="Strategy",
                enable_transaction_costs=False, transaction_cost_params=None,
                enable_turnover_constraint=False, max_turnover=None, 
                min_trade_threshold=None, min_position_weight=None):
    """
    Run backtest for a given portfolio with optional transaction costs and turnover constraints
    
    Args:
        portfolio_weights: Dictionary with dates as keys and weight Series as values
        monthly_returns: DataFrame with monthly returns
        strategy_name: Name for display
        enable_transaction_costs: Whether to deduct transaction costs
        transaction_cost_params: Dict with cost parameters
        enable_turnover_constraint: Whether to enforce turnover limit
        max_turnover: Maximum allowed turnover (if constraint enabled)
        min_trade_threshold: Minimum trade size (if constraint enabled)
        min_position_weight: Minimum position weight (if constraint enabled)
    
    Returns:
        equity_curve: Series with portfolio value over time
        strategy_returns: Series with monthly portfolio returns
        turnover_series: Series with monthly turnover
        transaction_costs_series: Series with monthly transaction costs
    """
    print(f"\n{'=' * 60}")
    print(f"📈 RUNNING {strategy_name.upper()} BACKTEST")
    if enable_transaction_costs:
        print(f"💰 Transaction costs: ENABLED")
    if enable_turnover_constraint:
        print(f"🔒 Turnover constraint: {max_turnover:.1%} max")
    print(f"{'=' * 60}")

    # List to store monthly portfolio returns
    strategy_returns = pd.Series(index=monthly_returns.index, dtype=float)
    turnover_series = pd.Series(index=monthly_returns.index, dtype=float)
    transaction_costs_series = pd.Series(index=monthly_returns.index, dtype=float)
    
    # Track previous weights for turnover calculation
    previous_weights = pd.Series(dtype=float)
    
    num_constrained = 0

    # Iterate over months with defined weights
    for date in sorted(portfolio_weights.keys()):
        next_month_idx = monthly_returns.index.get_indexer([date], method='bfill')[0] + 1

        if next_month_idx < len(monthly_returns):
            target_date = monthly_returns.index[next_month_idx]
            target_weights = portfolio_weights[date]  # What strategy wants (already normalized)
            
            # Ensure all assets have returns available (filter out missing)
            available_assets = target_weights.index.intersection(monthly_returns.columns)
            if len(available_assets) == 0:
                continue
            
            # Filter target weights to only available assets
            target_weights_filtered = target_weights[available_assets]
            # Normalize after filtering
            if target_weights_filtered.sum() > 0:
                target_weights_filtered = target_weights_filtered / target_weights_filtered.sum()
            
            # Now apply turnover constraint to the NORMALIZED filtered weights
            if enable_turnover_constraint and len(previous_weights) > 0 and max_turnover is not None:
                # Apply constraint based on DRIFTED weights vs normalized filtered targets
                constrained_weights, actual_turnover_constraint, was_constrained = apply_turnover_constraint(
                    old_weights=previous_weights,
                    new_weights=target_weights_filtered,
                    max_turnover=max_turnover,
                    min_trade_threshold=min_trade_threshold if min_trade_threshold is not None else 0.01,
                    min_position_weight=min_position_weight
                )
                if was_constrained:
                    num_constrained += 1
                weights_to_use = constrained_weights
            else:
                weights_to_use = target_weights_filtered
            
            # Get returns
            rets = monthly_returns.loc[target_date, weights_to_use.index]

            # Portfolio return for that month: sum(weight * return)
            gross_return = (weights_to_use * rets).sum()
            
            # Calculate transaction costs and turnover
            transaction_cost = 0.0
            turnover = 0.0
            
            # Always calculate turnover for reporting
            if len(previous_weights) > 0:
                turnover = calculate_turnover(previous_weights, weights_to_use)
            else:
                turnover = 1.0
            
            # Calculate transaction costs if enabled
            if enable_transaction_costs and transaction_cost_params is not None:
                if len(previous_weights) > 0:
                    # Align old and new weights
                    all_assets = previous_weights.index.union(weights_to_use.index)
                    old_aligned = previous_weights.reindex(all_assets, fill_value=0)
                    new_aligned = weights_to_use.reindex(all_assets, fill_value=0)
                    
                    trades = new_aligned - old_aligned
                    
                    # Calculate costs
                    transaction_cost, _ = calculate_transaction_costs(
                        trades=trades,
                        portfolio_value=transaction_cost_params['portfolio_value'],
                        bid_ask_spread=transaction_cost_params['bid_ask_spread'],
                        commission_per_trade=transaction_cost_params['commission_per_trade'],
                        commission_per_share=transaction_cost_params['commission_per_share'],
                        market_impact_factor=transaction_cost_params['market_impact_factor']
                    )
                else:
                    # First rebalance - assume starting from cash (100% turnover)
                    trades = weights_to_use
                    transaction_cost, _ = calculate_transaction_costs(
                        trades=trades,
                        portfolio_value=transaction_cost_params['portfolio_value'],
                        bid_ask_spread=transaction_cost_params['bid_ask_spread'],
                        commission_per_trade=transaction_cost_params['commission_per_trade'],
                        commission_per_share=transaction_cost_params['commission_per_share'],
                        market_impact_factor=transaction_cost_params['market_impact_factor']
                    )
            
            # Net return after transaction costs
            net_return = gross_return - transaction_cost
            
            strategy_returns[target_date] = net_return
            turnover_series[target_date] = turnover
            transaction_costs_series[target_date] = transaction_cost
            
            # Update previous weights (natural drift with returns)
            # Next month's starting weights = today's weights adjusted for returns
            previous_weights = weights_to_use * (1 + rets)
            if previous_weights.sum() > 0:
                previous_weights = previous_weights / previous_weights.sum()

    # Cleaning and Equity Curve calculation (starting from 100)
    strategy_returns = strategy_returns.dropna()
    turnover_series = turnover_series.dropna()
    transaction_costs_series = transaction_costs_series.dropna()
    
    if len(strategy_returns) == 0:
        print(f"❌ No valid returns for {strategy_name}")
        return pd.Series(), pd.Series(), pd.Series(), pd.Series()
    
    equity_curve = (1 + strategy_returns).cumprod() * 100

    # Basic Metrics Calculation
    total_return = (equity_curve.iloc[-1] / 100) - 1
    ann_return = (1 + total_return) ** (12 / len(strategy_returns)) - 1
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * (12 ** 0.5) if strategy_returns.std() > 0 else 0
    
    # Sortino Ratio (only penalizes downside volatility)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = (strategy_returns.mean() / downside_std) * (12 ** 0.5) if downside_std > 0 else 0
    
    # Drawdown Calculation
    rolling_peak = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_peak) / rolling_peak
    max_drawdown = drawdown_series.min()
    if hasattr(max_drawdown, 'item'):
        max_drawdown = max_drawdown.item()
    else:
        max_drawdown = float(max_drawdown)
    
    # Turnover metrics
    avg_monthly_turnover = turnover_series[turnover_series > 0].mean()
    total_transaction_costs = transaction_costs_series.sum()

    print(f"✓ Backtest done:")
    print(f"  - Total Return: {total_return:.2%}")
    print(f"  - Annual Return: {ann_return:.2%}")
    print(f"  - Sharpe Ratio: {sharpe:.2f}")
    print(f"  - Sortino Ratio: {sortino:.2f}")
    print(f"  - Max Drawdown: {max_drawdown:.2%}")
    
    if enable_transaction_costs:
        print(f"  - Avg Monthly Turnover: {avg_monthly_turnover:.2%}")
        print(f"  - Total Transaction Costs: {total_transaction_costs:.2%}")
        print(f"  - Cost per Rebalance: {total_transaction_costs / len(turnover_series[turnover_series > 0]):.4%}")
    
    if enable_turnover_constraint:
        print(f"  - Times Constrained: {num_constrained}/{len(portfolio_weights)} rebalances")

    return equity_curve, strategy_returns, turnover_series, transaction_costs_series


def run_backtest_silent(portfolio_weights, monthly_returns,
                        enable_transaction_costs=False, transaction_cost_params=None,
                        enable_turnover_constraint=False, max_turnover=None,
                        min_trade_threshold=None, min_position_weight=None):
    """
    Silent version of run_backtest for Monte Carlo simulations.
    Same logic, no print statements.
    
    Returns:
        equity_curve, strategy_returns
    """
    strategy_returns = pd.Series(index=monthly_returns.index, dtype=float)
    previous_weights = pd.Series(dtype=float)

    for date in sorted(portfolio_weights.keys()):
        next_month_idx = monthly_returns.index.get_indexer([date], method='bfill')[0] + 1

        if next_month_idx < len(monthly_returns):
            target_date = monthly_returns.index[next_month_idx]
            target_weights = portfolio_weights[date]

            available_assets = target_weights.index.intersection(monthly_returns.columns)
            if len(available_assets) == 0:
                continue

            target_weights_filtered = target_weights[available_assets]
            if target_weights_filtered.sum() > 0:
                target_weights_filtered = target_weights_filtered / target_weights_filtered.sum()

            if enable_turnover_constraint and len(previous_weights) > 0 and max_turnover is not None:
                constrained_weights, _, _ = apply_turnover_constraint(
                    old_weights=previous_weights,
                    new_weights=target_weights_filtered,
                    max_turnover=max_turnover,
                    min_trade_threshold=min_trade_threshold if min_trade_threshold is not None else 0.01,
                    min_position_weight=min_position_weight
                )
                weights_to_use = constrained_weights
            else:
                weights_to_use = target_weights_filtered

            rets = monthly_returns.loc[target_date, weights_to_use.index]
            gross_return = (weights_to_use * rets).sum()

            # Transaction costs
            transaction_cost = 0.0
            if enable_transaction_costs and transaction_cost_params is not None:
                if len(previous_weights) > 0:
                    all_assets = previous_weights.index.union(weights_to_use.index)
                    old_aligned = previous_weights.reindex(all_assets, fill_value=0)
                    new_aligned = weights_to_use.reindex(all_assets, fill_value=0)
                    trades = new_aligned - old_aligned
                    transaction_cost, _ = calculate_transaction_costs(
                        trades=trades,
                        portfolio_value=transaction_cost_params['portfolio_value'],
                        bid_ask_spread=transaction_cost_params['bid_ask_spread'],
                        commission_per_trade=transaction_cost_params['commission_per_trade'],
                        commission_per_share=transaction_cost_params['commission_per_share'],
                        market_impact_factor=transaction_cost_params['market_impact_factor']
                    )
                else:
                    trades = weights_to_use
                    transaction_cost, _ = calculate_transaction_costs(
                        trades=trades,
                        portfolio_value=transaction_cost_params['portfolio_value'],
                        bid_ask_spread=transaction_cost_params['bid_ask_spread'],
                        commission_per_trade=transaction_cost_params['commission_per_trade'],
                        commission_per_share=transaction_cost_params['commission_per_share'],
                        market_impact_factor=transaction_cost_params['market_impact_factor']
                    )

            net_return = gross_return - transaction_cost
            strategy_returns[target_date] = net_return

            previous_weights = weights_to_use * (1 + rets)
            if previous_weights.sum() > 0:
                previous_weights = previous_weights / previous_weights.sum()

    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) == 0:
        return pd.Series(), pd.Series()

    equity_curve = (1 + strategy_returns).cumprod() * 100
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
# EXCEL EXPORT (UPDATED WITH TURNOVER AND COSTS)
# =============================================================================

def export_to_excel(filename, equity_curve, portfolio_weights, benchmark=None,
                   turnover_series=None, transaction_costs_series=None):
    """
    Export results to Excel file
    
    Args:
        filename: Output filename
        equity_curve: Portfolio equity curve Series
        portfolio_weights: Dictionary of portfolio weights
        benchmark: Optional benchmark Series
        turnover_series: Optional Series with monthly turnover
        transaction_costs_series: Optional Series with monthly transaction costs
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
            
            # Sheet 4: Turnover and Transaction Costs
            if turnover_series is not None and transaction_costs_series is not None:
                costs_df = pd.DataFrame({
                    'Date': turnover_series.index,
                    'Turnover': turnover_series.values,
                    'Turnover_Pct': (turnover_series.values * 100).round(2),
                    'Transaction_Cost': transaction_costs_series.values,
                    'Transaction_Cost_Bps': (transaction_costs_series.values * 10000).round(2)
                })
                costs_df.to_excel(writer, sheet_name='Turnover_Costs', index=False)
                
                # Summary statistics
                summary_data = {
                    'Metric': [
                        'Average Monthly Turnover',
                        'Maximum Monthly Turnover',
                        'Total Transaction Costs',
                        'Average Cost per Rebalance',
                        'Number of Rebalances'
                    ],
                    'Value': [
                        f"{turnover_series[turnover_series > 0].mean():.2%}",
                        f"{turnover_series.max():.2%}",
                        f"{transaction_costs_series.sum():.4%}",
                        f"{transaction_costs_series[transaction_costs_series > 0].mean():.4%}",
                        len(turnover_series[turnover_series > 0])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Cost_Summary', index=False)

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
    
    # Filter out positions below 5% minimum (matching MIN_STOCK_WEIGHT constraint)
    # Positions should all be >= 5%, but turnover constraints might create smaller positions
    MIN_WEIGHT_THRESHOLD = 0.05  # 5% minimum
    significant_weights = latest_weights[latest_weights >= MIN_WEIGHT_THRESHOLD]

    # Build message
    message = f"🚀 *PORTFOLIO REBALANCING ALERT*\n"
    message += f"📅 Rebalance Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    message += f"📅 Alert Sent: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    message += f"📈 *STOCKS ({len(significant_weights)} positions):*\n"
    for asset, weight in significant_weights.items():
        weight_pct = weight * 100
        # Use .0f for integer percentages (matching ROUND_TO_INTEGER = True)
        message += f"• {asset}: {weight_pct:.0f}%\n"
    
    # Show total allocation
    total_allocation = significant_weights.sum() * 100
    message += f"\n📊 *Total Allocation: {total_allocation:.0f}%*"
    
    # Warn if there are positions below minimum
    small_positions = latest_weights[latest_weights < MIN_WEIGHT_THRESHOLD]
    if len(small_positions) > 0:
        message += f"\n\n⚠️ *Warning: {len(small_positions)} position(s) below 5% minimum*"
        for asset, weight in small_positions.items():
            message += f"\n• {asset}: {weight*100:.1f}%"
    
    message += f"\n\n🔄 _Please update your brokerage account positions._"

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
# PLOTTING (UPDATED WITH TURNOVER)
# =============================================================================

def plot_performance(equity_curve, benchmark=None, title="Portfolio Performance",
                    turnover_series=None):
    """
    Plot performance with equity curve, drawdown, and optional turnover subplot
    
    Args:
        equity_curve: Portfolio equity curve Series
        benchmark: Optional benchmark Series
        title: Plot title
        turnover_series: Optional Series with monthly turnover
    """
    print(f"\n📊 Generating performance plot...")

    # Calculate drawdown series
    rolling_peak = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_peak) / rolling_peak * 100  # in %

    # Benchmark drawdown (if available)
    bm_equity_aligned = None
    bm_drawdown_series = None
    ann_return_bm = None
    
    if benchmark is not None and not benchmark.empty:
        benchmark_clean = benchmark.dropna()
        if len(benchmark_clean) > 0:
            benchmark_returns_aligned = benchmark_clean.reindex(equity_curve.index).dropna()
            if len(benchmark_returns_aligned) > 0:
                bm_equity_aligned = (1 + benchmark_returns_aligned).cumprod() * 100
                bm_peak = bm_equity_aligned.cummax()
                bm_drawdown_series = (bm_equity_aligned - bm_peak) / bm_peak * 100
                
                bm_final_value = bm_equity_aligned.iloc[-1]
                if hasattr(bm_final_value, 'item'):
                    bm_final_value = bm_final_value.item()
                else:
                    bm_final_value = float(bm_final_value)
                total_return_bm = (bm_final_value / 100) - 1
                years = len(benchmark_returns_aligned) / 12
                ann_return_bm = (1 + total_return_bm) ** (1 / years) - 1

    # Determine subplot layout
    has_turnover = turnover_series is not None and len(turnover_series) > 0
    if has_turnover:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14),
                                              gridspec_kw={'height_ratios': [3, 1.5, 1]})
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11),
                                         gridspec_kw={'height_ratios': [3, 1.5]})

    # =========================================================================
    # Panel 1: Equity Curve
    # =========================================================================
    final_value = equity_curve.iloc[-1]
    if hasattr(final_value, 'item'):
        final_value = final_value.item()
    else:
        final_value = float(final_value)
    total_return = (final_value / 100) - 1
    ann_return = (1 + total_return) ** (12 / len(equity_curve)) - 1

    ax1.plot(equity_curve.index, equity_curve.values,
             label=f'Portfolio (Ann: {ann_return:.1%})', lw=2.5, color='#2E86AB')

    if bm_equity_aligned is not None:
        ax1.plot(bm_equity_aligned.index, bm_equity_aligned.values,
                label=f'Benchmark (Ann: {ann_return_bm:.1%})',
                linestyle='--', alpha=0.7, lw=2, color='#A23B72')

    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (Base 100)', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Drawdown
    # =========================================================================
    dd_min = drawdown_series.min()
    if hasattr(dd_min, 'item'):
        dd_min = dd_min.item()
    else:
        dd_min = float(dd_min)
    ax2.fill_between(drawdown_series.index, drawdown_series.values, 0,
                     color='#E74C3C', alpha=0.4, label=f'Portfolio (Max: {dd_min:.1f}%)')
    ax2.plot(drawdown_series.index, drawdown_series.values, color='#E74C3C', lw=1.2)

    if bm_drawdown_series is not None:
        bm_dd_min = bm_drawdown_series.min()
        if hasattr(bm_dd_min, 'item'):
            bm_dd_min = bm_dd_min.item()
        else:
            bm_dd_min = float(bm_dd_min)
        ax2.plot(bm_drawdown_series.index, bm_drawdown_series.values,
                color='#A23B72', lw=1.2, linestyle='--', alpha=0.7,
                label=f'Benchmark (Max: {bm_dd_min:.1f}%)')

    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Drawdown from Peak', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(drawdown_series.min() * 1.15, -5), 2)

    # =========================================================================
    # Panel 3: Turnover (if provided)
    # =========================================================================
    if has_turnover:
        turnover_pct = turnover_series * 100
        ax3.bar(turnover_pct.index, turnover_pct.values,
               color='#F77F00', alpha=0.7, width=20)
        ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Limit')
        ax3.set_ylabel('Turnover (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('Monthly Portfolio Turnover', fontsize=14)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax2.set_xlabel('Date', fontsize=12)

    plt.tight_layout()

    # Save figure
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'portfolio_performance.png'")

    plt.show()
