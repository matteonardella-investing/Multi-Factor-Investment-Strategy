#!/usr/bin/env python3
"""
PORTFOLIO MANAGER - STOCKS ONLY
Simplified version focusing on stock selection strategy
"""

import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    START_DATE, END_DATE, BENCHMARK_TICKER,
    REBALANCE_FREQUENCY,
    SECTOR_LIMITS, MIN_STOCK_WEIGHT, ALPHA_WINDOW, MOMENTUM_LOOKBACK,
    ROUND_TO_INTEGER,
    TELEGRAM_ENABLED, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    EXCEL_OUTPUT_FILE, PLOT_ENABLED
)

# Import strategy modules
from stock_picker import run_stock_strategy

# Import utility functions
from portfolio_utils import (
    run_backtest, get_benchmark,
    export_to_excel, send_telegram_alert, plot_performance
)


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("🚀 PORTFOLIO MANAGER: STOCKS ONLY")
    print("=" * 80)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"📊 Strategy: Stock Selection with Fama-French Alpha Filter")
    print(f"🔄 Rebalancing: Every {REBALANCE_FREQUENCY} month(s)")
    print("=" * 80)

    # =============================================================================
    # STEP 1: RUN STOCK STRATEGY
    # =============================================================================
    
    portfolio_weights, monthly_returns = run_stock_strategy(
        start_date=START_DATE,
        end_date=END_DATE,
        sector_limits=SECTOR_LIMITS,
        min_weight=MIN_STOCK_WEIGHT,
        alpha_window=ALPHA_WINDOW,
        momentum_lookback=MOMENTUM_LOOKBACK,
        round_to_integer=ROUND_TO_INTEGER
    )

    if not portfolio_weights:
        print("❌ Stock strategy failed - no weights generated")
        return

    # =============================================================================
    # STEP 2: RUN BACKTEST
    # =============================================================================
    
    equity_curve, strategy_returns = run_backtest(
        portfolio_weights, 
        monthly_returns, 
        "Stock Portfolio"
    )

    # Get benchmark
    benchmark_returns = get_benchmark(BENCHMARK_TICKER, START_DATE, END_DATE)

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

    # =============================================================================
    # STEP 4: EXPORT TO EXCEL
    # =============================================================================
    
    print(f"\n{'=' * 80}")
    export_to_excel(
        filename=EXCEL_OUTPUT_FILE,
        equity_curve=equity_curve,
        portfolio_weights=portfolio_weights,
        benchmark=benchmark_returns
    )

    # =============================================================================
    # STEP 5: PLOT RESULTS
    # =============================================================================
    
    if PLOT_ENABLED:
        plot_performance(
            equity_curve=equity_curve,
            benchmark=benchmark_returns,
            title="Stock Portfolio Performance"
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
    print("✅ PORTFOLIO MANAGER COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
