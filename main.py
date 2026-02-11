#!/usr/bin/env python3
"""
PORTFOLIO MANAGER - STOCKS ONLY
Enhanced version with transaction costs and turnover constraints
"""

import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    START_DATE, END_DATE, BENCHMARK_TICKER,
    REBALANCE_FREQUENCY,
    SECTOR_LIMITS, MIN_STOCK_WEIGHT, ALPHA_WINDOW, MOMENTUM_LOOKBACK,
    ROUND_TO_INTEGER,
    ENABLE_TRANSACTION_COSTS, BID_ASK_SPREAD, COMMISSION_PER_TRADE, 
    COMMISSION_PER_SHARE, MARKET_IMPACT_FACTOR, ASSUMED_PORTFOLIO_VALUE,
    ENABLE_TURNOVER_CONSTRAINT, MAX_MONTHLY_TURNOVER, MIN_TRADE_THRESHOLD,
    MONTE_CARLO_ENABLED, MC_N_SIMULATIONS, MC_HORIZON_MONTHS,
    MC_CONFIDENCE_LEVELS, MC_RANDOM_SEED, MC_SAMPLE_PATHS_PLOT,
    MC_PLOT_FILE, MC_EXCEL_FILE,
    ROBUSTNESS_MC_ENABLED, ROBUSTNESS_N_SIMULATIONS, ROBUSTNESS_NOISE_SCALE,
    ROBUSTNESS_RANDOM_SEED, ROBUSTNESS_PLOT_FILE, ROBUSTNESS_EXCEL_FILE,
    TELEGRAM_ENABLED, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    EXCEL_OUTPUT_FILE, PLOT_ENABLED
)

# Import strategy modules
from stock_picker import run_stock_strategy

# Import utility functions
from portfolio_utils import (
    run_backtest, get_benchmark, apply_turnover_constraint,
    export_to_excel, send_telegram_alert, plot_performance
)

# Import Monte Carlo module
from monte_carlo import (
    run_monte_carlo, print_monte_carlo_report,
    plot_monte_carlo, export_monte_carlo_to_excel,
    run_robustness_monte_carlo, print_robustness_report,
    plot_robustness_monte_carlo, export_robustness_to_excel
)


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("🚀 PORTFOLIO MANAGER: STOCKS ONLY (v2.0)")
    print("=" * 80)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"📊 Strategy: Stock Selection with Fama-French Alpha Filter")
    print(f"🔄 Rebalancing: Every {REBALANCE_FREQUENCY} month(s)")
    
    if ENABLE_TRANSACTION_COSTS:
        print(f"\n💰 TRANSACTION COSTS: ENABLED")
        print(f"  - Bid-Ask Spread: {BID_ASK_SPREAD*10000:.1f} bps")
        print(f"  - Commission per Trade: ${COMMISSION_PER_TRADE:.2f}")
        print(f"  - Commission per Share: ${COMMISSION_PER_SHARE:.4f}")
        print(f"  - Market Impact Factor: {MARKET_IMPACT_FACTOR*10000:.1f} bps")
    
    if ENABLE_TURNOVER_CONSTRAINT:
        print(f"\n🔒 TURNOVER CONSTRAINT: ENABLED")
        print(f"  - Maximum Monthly Turnover: {MAX_MONTHLY_TURNOVER:.1%}")
        print(f"  - Minimum Trade Threshold: {MIN_TRADE_THRESHOLD:.1%}")
    
    if MONTE_CARLO_ENABLED:
        print(f"\n🎲 BOOTSTRAP MONTE CARLO: ENABLED")
        print(f"  - Simulations: {MC_N_SIMULATIONS:,}")
        print(f"  - Horizon: {MC_HORIZON_MONTHS} months ({MC_HORIZON_MONTHS / 12:.0f} years)")
    
    if ROBUSTNESS_MC_ENABLED:
        print(f"\n🔬 ROBUSTNESS MONTE CARLO: ENABLED")
        print(f"  - Simulations: {ROBUSTNESS_N_SIMULATIONS}")
        print(f"  - Noise Scale: {ROBUSTNESS_NOISE_SCALE:.0%}")
    
    print("=" * 80)

    # =============================================================================
    # STEP 1: RUN STOCK STRATEGY
    # =============================================================================
    
    intermediate_data = None
    
    if ROBUSTNESS_MC_ENABLED:
        # Request intermediate data for robustness MC
        portfolio_weights_raw, monthly_returns, intermediate_data = run_stock_strategy(
            start_date=START_DATE,
            end_date=END_DATE,
            sector_limits=SECTOR_LIMITS,
            min_weight=MIN_STOCK_WEIGHT,
            alpha_window=ALPHA_WINDOW,
            momentum_lookback=MOMENTUM_LOOKBACK,
            round_to_integer=ROUND_TO_INTEGER,
            return_intermediate_data=True
        )
    else:
        portfolio_weights_raw, monthly_returns = run_stock_strategy(
            start_date=START_DATE,
            end_date=END_DATE,
            sector_limits=SECTOR_LIMITS,
            min_weight=MIN_STOCK_WEIGHT,
            alpha_window=ALPHA_WINDOW,
            momentum_lookback=MOMENTUM_LOOKBACK,
            round_to_integer=ROUND_TO_INTEGER
        )

    if not portfolio_weights_raw:
        print("❌ Stock strategy failed - no weights generated")
        return

    # =============================================================================
    # STEP 1.5: NOTE ON TURNOVER CONSTRAINTS
    # =============================================================================
    
    # NOTE: Turnover constraints are now applied DURING the backtest, not here.
    # This is because we need to account for natural portfolio drift between rebalances.
    # The constraint is applied in run_backtest() based on actual drifted weights.
    
    if ENABLE_TURNOVER_CONSTRAINT:
        print(f"\n{'=' * 80}")
        print(f"🔒 TURNOVER CONSTRAINT ENABLED")
        print(f"  Maximum monthly turnover: {MAX_MONTHLY_TURNOVER:.1%}")
        print(f"  Minimum trade threshold: {MIN_TRADE_THRESHOLD:.1%}")
        print(f"  Minimum position weight: {MIN_STOCK_WEIGHT:.1%}")
        print(f"  (Applied during backtest accounting for drift)")
        print(f"{'=' * 80}")
    
    # Use raw weights - constraints will be applied in backtest
    portfolio_weights = portfolio_weights_raw

    # =============================================================================
    # STEP 2: RUN BACKTEST WITH TRANSACTION COSTS
    # =============================================================================
    
    # Prepare transaction cost parameters
    transaction_cost_params = None
    if ENABLE_TRANSACTION_COSTS:
        transaction_cost_params = {
            'portfolio_value': ASSUMED_PORTFOLIO_VALUE,
            'bid_ask_spread': BID_ASK_SPREAD,
            'commission_per_trade': COMMISSION_PER_TRADE,
            'commission_per_share': COMMISSION_PER_SHARE,
            'market_impact_factor': MARKET_IMPACT_FACTOR
        }
    
    equity_curve, strategy_returns, turnover_series, transaction_costs_series = run_backtest(
        portfolio_weights=portfolio_weights,
        monthly_returns=monthly_returns,
        strategy_name="Stock Portfolio",
        enable_transaction_costs=ENABLE_TRANSACTION_COSTS,
        transaction_cost_params=transaction_cost_params,
        enable_turnover_constraint=ENABLE_TURNOVER_CONSTRAINT,
        max_turnover=MAX_MONTHLY_TURNOVER if ENABLE_TURNOVER_CONSTRAINT else None,
        min_trade_threshold=MIN_TRADE_THRESHOLD if ENABLE_TURNOVER_CONSTRAINT else None,
        min_position_weight=MIN_STOCK_WEIGHT if ENABLE_TURNOVER_CONSTRAINT else None
    )

    # Get benchmark
    benchmark_returns = get_benchmark(BENCHMARK_TICKER, START_DATE, END_DATE)

    # =============================================================================
    # STEP 2.5: MONTE CARLO SIMULATION
    # =============================================================================
    
    mc_results = None
    
    if MONTE_CARLO_ENABLED and len(strategy_returns) > 12:
        print(f"\n{'=' * 80}")
        print(f"🎲 MONTE CARLO SIMULATION")
        print(f"{'=' * 80}")
        print(f"  Simulations: {MC_N_SIMULATIONS:,}")
        print(f"  Horizon: {MC_HORIZON_MONTHS} months ({MC_HORIZON_MONTHS / 12:.1f} years)")
        print(f"  Seed: {MC_RANDOM_SEED if MC_RANDOM_SEED else 'Random'}")
        
        mc_results = run_monte_carlo(
            strategy_returns=strategy_returns,
            n_simulations=MC_N_SIMULATIONS,
            n_months=MC_HORIZON_MONTHS,
            initial_value=100,
            confidence_levels=MC_CONFIDENCE_LEVELS,
            random_seed=MC_RANDOM_SEED
        )
        
        # Print comprehensive report
        print_monte_carlo_report(mc_results)
        
        # Generate plots
        plot_monte_carlo(
            mc_results=mc_results,
            title=f"Monte Carlo Simulation - Stock Portfolio ({MC_N_SIMULATIONS:,} paths, {MC_HORIZON_MONTHS // 12}Y horizon)",
            n_sample_paths=MC_SAMPLE_PATHS_PLOT,
            save_path=MC_PLOT_FILE
        )
        
        # Export to Excel
        export_monte_carlo_to_excel(mc_results, filename=MC_EXCEL_FILE)
    
    elif MONTE_CARLO_ENABLED and len(strategy_returns) <= 12:
        print(f"\n⚠️  Monte Carlo skipped: need >12 months of returns, have {len(strategy_returns)}")

    # =============================================================================
    # STEP 2.6: ROBUSTNESS MONTE CARLO (Approach 3)
    # =============================================================================
    
    robustness_results = None
    original_ann_return = None
    
    if ROBUSTNESS_MC_ENABLED and intermediate_data is not None:
        print(f"\n{'=' * 80}")
        print(f"🔬 ROBUSTNESS MONTE CARLO (Weight Perturbation)")
        print(f"{'=' * 80}")
        print(f"  Simulations: {ROBUSTNESS_N_SIMULATIONS}")
        print(f"  Noise scale: {ROBUSTNESS_NOISE_SCALE:.0%}")
        print(f"  Seed: {ROBUSTNESS_RANDOM_SEED if ROBUSTNESS_RANDOM_SEED else 'Random'}")
        
        # Calculate original annualized return for comparison
        if len(equity_curve) > 0:
            orig_total = (equity_curve.iloc[-1] / 100) - 1
            original_ann_return = (1 + orig_total) ** (12 / len(strategy_returns)) - 1
        
        robustness_results = run_robustness_monte_carlo(
            intermediate_data=intermediate_data,
            monthly_returns=monthly_returns,
            sector_limits=SECTOR_LIMITS,
            min_weight=MIN_STOCK_WEIGHT,
            n_simulations=ROBUSTNESS_N_SIMULATIONS,
            noise_scale=ROBUSTNESS_NOISE_SCALE,
            enable_transaction_costs=ENABLE_TRANSACTION_COSTS,
            transaction_cost_params=transaction_cost_params,
            enable_turnover_constraint=ENABLE_TURNOVER_CONSTRAINT,
            max_turnover=MAX_MONTHLY_TURNOVER if ENABLE_TURNOVER_CONSTRAINT else None,
            min_trade_threshold=MIN_TRADE_THRESHOLD if ENABLE_TURNOVER_CONSTRAINT else None,
            min_position_weight=MIN_STOCK_WEIGHT if ENABLE_TURNOVER_CONSTRAINT else None,
            random_seed=ROBUSTNESS_RANDOM_SEED
        )
        
        if robustness_results is not None:
            print_robustness_report(robustness_results, original_ann_return=original_ann_return)
            
            plot_robustness_monte_carlo(
                robustness_results=robustness_results,
                original_equity_curve=equity_curve,
                original_ann_return=original_ann_return,
                title=f"Robustness MC - {ROBUSTNESS_N_SIMULATIONS} sims, {ROBUSTNESS_NOISE_SCALE:.0%} noise",
                save_path=ROBUSTNESS_PLOT_FILE
            )
            
            export_robustness_to_excel(
                robustness_results,
                original_ann_return=original_ann_return,
                filename=ROBUSTNESS_EXCEL_FILE
            )
    
    elif ROBUSTNESS_MC_ENABLED and intermediate_data is None:
        print(f"\n⚠️  Robustness MC skipped: intermediate data not available")

    # =============================================================================
    # STEP 3: PERFORMANCE REPORT
    # =============================================================================
    
    print(f"\n{'=' * 80}")
    print(f"🏆 FINAL PERFORMANCE REPORT")
    print(f"{'=' * 80}")

    if len(equity_curve) > 0:
        total_ret = (equity_curve.iloc[-1] / 100) - 1
        ann_ret = (1 + total_ret) ** (12 / len(equity_curve)) - 1
        
        print(f"\nPortfolio Performance:")
        print(f"  Total Return: {total_ret:.2%}")
        print(f"  Annual Return: {ann_ret:.2%}")
        
        # Drawdown metrics
        rolling_peak = equity_curve.cummax()
        drawdown_series = (equity_curve - rolling_peak) / rolling_peak
        max_dd = drawdown_series.min()
        if hasattr(max_dd, 'item'):
            max_dd = max_dd.item()
        else:
            max_dd = float(max_dd)
        
        # Find max drawdown date and recovery
        max_dd_date = drawdown_series.idxmin()
        # Find the peak before the max drawdown
        peak_before_dd = equity_curve.loc[:max_dd_date].idxmax()
        # Check if recovered
        post_dd = equity_curve.loc[max_dd_date:]
        peak_val = equity_curve.loc[peak_before_dd]
        if hasattr(peak_val, 'item'):
            peak_val = peak_val.item()
        else:
            peak_val = float(peak_val)
        recovered = post_dd[post_dd >= peak_val]
        
        sharpe_val = (strategy_returns.mean() / strategy_returns.std()) * (12 ** 0.5) if strategy_returns.std() > 0 else 0
        downside_rets = strategy_returns[strategy_returns < 0]
        downside_std = downside_rets.std() if len(downside_rets) > 0 else 0
        sortino_val = (strategy_returns.mean() / downside_std) * (12 ** 0.5) if downside_std > 0 else 0
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        print(f"  Sharpe Ratio: {sharpe_val:.2f}")
        print(f"  Sortino Ratio: {sortino_val:.2f}")
        print(f"  Max Drawdown: {max_dd:.2%}")
        print(f"  Max Drawdown Date: {max_dd_date.strftime('%Y-%m')}")
        print(f"  Peak Before DD: {peak_before_dd.strftime('%Y-%m')}")
        if len(recovered) > 0:
            recovery_date = recovered.index[0]
            recovery_months = (recovery_date.year - max_dd_date.year) * 12 + (recovery_date.month - max_dd_date.month)
            print(f"  Recovery Date: {recovery_date.strftime('%Y-%m')} ({recovery_months} months)")
        else:
            print(f"  Recovery: Not yet recovered")
        print(f"  Calmar Ratio: {calmar:.2f}")
        
        # Transaction cost impact
        if ENABLE_TRANSACTION_COSTS and len(transaction_costs_series) > 0:
            total_costs = transaction_costs_series.sum()
            avg_monthly_turnover = turnover_series[turnover_series > 0].mean() if len(turnover_series) > 0 else 0
            
            print(f"\nTransaction Costs Analysis:")
            print(f"  Total Costs: {total_costs:.2%}")
            print(f"  Cost Drag on Returns: {total_costs:.2%}")
            print(f"  Average Monthly Turnover: {avg_monthly_turnover:.1%}")
            print(f"  Number of Rebalances: {len(turnover_series[turnover_series > 0])}")

        # Benchmark comparison
        if not benchmark_returns.empty:
            # Align benchmark returns with portfolio dates
            bm_returns_aligned = benchmark_returns.reindex(equity_curve.index).dropna()
            
            if len(bm_returns_aligned) > 0:
                # Recalculate equity starting from 100
                bm_equity_aligned = (1 + bm_returns_aligned).cumprod() * 100
                
                # Extract scalar values properly
                bm_final = bm_equity_aligned.iloc[-1]
                if hasattr(bm_final, 'item'):
                    bm_final = bm_final.item()
                else:
                    bm_final = float(bm_final)
                
                bm_total = (bm_final / 100) - 1
                years = len(bm_returns_aligned) / 12
                bm_ann = (1 + bm_total) ** (1 / years) - 1
                
                portfolio_final = equity_curve.iloc[-1]
                if hasattr(portfolio_final, 'item'):
                    portfolio_final = portfolio_final.item()
                else:
                    portfolio_final = float(portfolio_final)
                
                portfolio_total = (portfolio_final / 100) - 1
                portfolio_ann = (1 + portfolio_total) ** (1 / years) - 1
                alpha = portfolio_ann - bm_ann
                
                print(f"\nBenchmark ({BENCHMARK_TICKER}):")
                print(f"  Annual Return: {bm_ann:.2%}")
                print(f"\n🎯 Alpha Generated: {alpha:.2%}")

        # Show latest portfolio composition
        if portfolio_weights:
            latest_date = max(portfolio_weights.keys())
            latest_weights = portfolio_weights[latest_date].sort_values(ascending=False)
            
            print(f"\n📊 Current Portfolio ({latest_date.strftime('%Y-%m')}):")
            print(f"  - Number of positions: {len(latest_weights)}")
            print(f"  - Top 5 holdings:")
            for i, (ticker, weight) in enumerate(latest_weights.head(5).items(), 1):
                print(f"    {i}. {ticker}: {weight*100:.0f}%")
        
        # Monte Carlo summary (if available)
        if mc_results is not None:
            mc_stats = mc_results['stats']
            years = mc_stats['n_months'] / 12
            print(f"\n🎲 Monte Carlo Projection ({years:.0f}-Year, {mc_stats['n_simulations']:,} sims):")
            print(f"  Median terminal value: {mc_stats['median_terminal']:.2f}")
            print(f"  Probability of loss: {mc_stats['prob_loss']:.1%}")
            print(f"  5th-95th percentile range: {mc_stats['worst_case_5']:.1f} - {mc_stats['best_case_95']:.1f}")
            print(f"  Median max drawdown: {mc_stats['median_max_drawdown']:.1%}")
        
        # Robustness MC summary (if available)
        if robustness_results is not None:
            rb_stats = robustness_results['stats']
            print(f"\n🔬 Robustness MC ({rb_stats['n_successful']} sims, {rb_stats['noise_scale']:.0%} noise):")
            print(f"  Median perturbed annual return: {rb_stats['median_ann_return']:.2%}")
            print(f"  % sims with positive return: {rb_stats['pct_positive_alpha']:.1f}%")
            print(f"  % sims with >15% annual: {rb_stats['pct_above_15pct']:.1f}%")
            print(f"  5th-95th return range: {rb_stats['p5_ann_return']:.1%} - {rb_stats['p95_ann_return']:.1%}")

    # =============================================================================
    # STEP 4: EXPORT TO EXCEL
    # =============================================================================
    
    print(f"\n{'=' * 80}")
    export_to_excel(
        filename=EXCEL_OUTPUT_FILE,
        equity_curve=equity_curve,
        portfolio_weights=portfolio_weights,
        benchmark=benchmark_returns,
        turnover_series=turnover_series if ENABLE_TURNOVER_CONSTRAINT or ENABLE_TRANSACTION_COSTS else None,
        transaction_costs_series=transaction_costs_series if ENABLE_TRANSACTION_COSTS else None
    )

    # =============================================================================
    # STEP 5: PLOT RESULTS
    # =============================================================================
    
    if PLOT_ENABLED:
        plot_performance(
            equity_curve=equity_curve,
            benchmark=benchmark_returns,
            title="Stock Portfolio Performance (with Transaction Costs)" if ENABLE_TRANSACTION_COSTS else "Stock Portfolio Performance",
            turnover_series=turnover_series if ENABLE_TURNOVER_CONSTRAINT or ENABLE_TRANSACTION_COSTS else None
        )

    # =============================================================================
    # STEP 6: SEND TELEGRAM ALERT
    # =============================================================================
    
    if TELEGRAM_ENABLED and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        print(f"\n{'=' * 80}")
        send_telegram_alert(
            token=TELEGRAM_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
            portfolio_weights=portfolio_weights
        )

    print(f"\n{'=' * 80}")
    print("✅ STRATEGY COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
