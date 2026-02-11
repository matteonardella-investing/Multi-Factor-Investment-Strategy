#!/usr/bin/env python3
"""
MONTE CARLO SIMULATION MODULE
Approach 1: Bootstrap (resample historical returns)
Approach 2: Robustness (perturb stock selection to test strategy sensitivity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import sys


# =============================================================================
# APPROACH 1: BOOTSTRAP MONTE CARLO
# =============================================================================

def run_monte_carlo(strategy_returns, n_simulations=10000, n_months=60,
                    initial_value=100, confidence_levels=None, random_seed=None):
    if confidence_levels is None:
        confidence_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    if random_seed is not None:
        np.random.seed(random_seed)

    returns_clean = strategy_returns.dropna().values
    n_historical = len(returns_clean)

    print(f"\n  Historical return statistics:")
    print(f"     - Observations: {n_historical} months")
    print(f"     - Mean monthly return: {np.mean(returns_clean):.4%}")
    print(f"     - Std monthly return: {np.std(returns_clean):.4%}")
    print(f"     - Skewness: {_calc_skewness(returns_clean):.3f}")
    print(f"     - Kurtosis (excess): {_calc_kurtosis(returns_clean):.3f}")
    print(f"     - Worst month: {np.min(returns_clean):.4%}")
    print(f"     - Best month: {np.max(returns_clean):.4%}")

    random_indices = np.random.randint(0, n_historical, size=(n_simulations, n_months))
    sampled_returns = returns_clean[random_indices]
    growth_factors = 1 + sampled_returns
    cumulative_growth = np.cumprod(growth_factors, axis=1)
    equity_paths = np.hstack([
        np.full((n_simulations, 1), initial_value),
        initial_value * cumulative_growth
    ])

    percentile_paths = pd.DataFrame(index=range(n_months + 1))
    for level in confidence_levels:
        percentile_paths[f'p{int(level * 100)}'] = np.percentile(equity_paths, level * 100, axis=0)

    terminal_values = equity_paths[:, -1]
    drawdown_distribution = np.array([_calc_max_drawdown_path(equity_paths[i, :]) for i in range(n_simulations)])

    monthly_stats = pd.DataFrame(index=range(n_months + 1))
    monthly_stats['mean'] = np.mean(equity_paths, axis=0)
    monthly_stats['std'] = np.std(equity_paths, axis=0)
    monthly_stats['min'] = np.min(equity_paths, axis=0)
    monthly_stats['max'] = np.max(equity_paths, axis=0)
    monthly_stats['prob_loss'] = np.mean(equity_paths < initial_value, axis=0)

    median_terminal = np.median(terminal_values)
    years = n_months / 12
    median_ann_return = (median_terminal / initial_value) ** (1 / years) - 1
    mean_terminal = np.mean(terminal_values)
    mean_ann_return = (mean_terminal / initial_value) ** (1 / years) - 1

    stats = {
        'mean_terminal': mean_terminal, 'median_terminal': median_terminal,
        'std_terminal': np.std(terminal_values),
        'mean_ann_return': mean_ann_return, 'median_ann_return': median_ann_return,
        'prob_loss': np.mean(terminal_values < initial_value),
        'prob_double': np.mean(terminal_values >= initial_value * 2),
        'prob_below_80': np.mean(terminal_values < initial_value * 0.80),
        'var_5pct': np.percentile(terminal_values, 5),
        'var_1pct': np.percentile(terminal_values, 1),
        'cvar_5pct': np.mean(terminal_values[terminal_values <= np.percentile(terminal_values, 5)]),
        'cvar_1pct': np.mean(terminal_values[terminal_values <= np.percentile(terminal_values, 1)]),
        'median_max_drawdown': np.median(drawdown_distribution),
        'mean_max_drawdown': np.mean(drawdown_distribution),
        'worst_max_drawdown_95': np.percentile(drawdown_distribution, 5),
        'best_case_95': np.percentile(terminal_values, 95),
        'worst_case_5': np.percentile(terminal_values, 5),
        'n_simulations': n_simulations, 'n_months': n_months, 'n_historical_months': n_historical,
    }

    return {
        'equity_paths': equity_paths, 'percentile_paths': percentile_paths,
        'terminal_values': terminal_values, 'stats': stats,
        'monthly_stats': monthly_stats, 'drawdown_distribution': drawdown_distribution,
        'confidence_levels': confidence_levels,
    }


# =============================================================================
# APPROACH 2: ROBUSTNESS MONTE CARLO (Weight Perturbation)
# =============================================================================

def run_robustness_monte_carlo(intermediate_data, monthly_returns,
                                sector_limits, min_weight=0.05,
                                n_simulations=200, noise_scale=0.20,
                                enable_transaction_costs=False,
                                transaction_cost_params=None,
                                enable_turnover_constraint=False,
                                max_turnover=None, min_trade_threshold=None,
                                min_position_weight=None, random_seed=None):
    from stock_picker import calculate_weights_silent
    from portfolio_utils import run_backtest_silent

    if random_seed is not None:
        np.random.seed(random_seed)

    dict_alpha_plus = intermediate_data['dict_alpha_plus']
    prices_daily = intermediate_data['prices_daily']
    momentum_original = intermediate_data['momentum']
    sector_mapping = intermediate_data['sector_mapping']

    print(f"\n  Perturbation settings:")
    print(f"     - Momentum noise scale: {noise_scale:.0%}")
    print(f"     - Alpha ranking shuffle: proportional to noise")
    print(f"     - Simulations: {n_simulations}")

    all_equity_curves = []
    all_annual_returns = []
    all_total_returns = []
    all_sharpe_ratios = []
    failed_sims = 0

    for sim in range(n_simulations):
        if (sim + 1) % 50 == 0 or sim == 0:
            pct_done = (sim + 1) / n_simulations * 100
            print(f"     Simulation {sim + 1}/{n_simulations} ({pct_done:.0f}%)...", end='')
            sys.stdout.flush()

        try:
            # PERTURBATION 1: Add noise to momentum scores
            noise = np.random.normal(0, noise_scale, momentum_original.shape)
            momentum_perturbed = momentum_original * (1 + noise)

            # PERTURBATION 2: Shuffle alpha rankings slightly
            dict_alpha_perturbed = {}
            swap_prob = min(noise_scale, 0.5)
            for date, tickers in dict_alpha_plus.items():
                perturbed_list = list(tickers)
                n_swaps = max(1, int(len(perturbed_list) * swap_prob * 0.3))
                for _ in range(n_swaps):
                    if len(perturbed_list) < 2:
                        break
                    idx = np.random.randint(0, len(perturbed_list) - 1)
                    perturbed_list[idx], perturbed_list[idx + 1] = perturbed_list[idx + 1], perturbed_list[idx]
                dict_alpha_perturbed[date] = perturbed_list

            # RECALCULATE WEIGHTS
            perturbed_weights = calculate_weights_silent(
                dict_alpha_perturbed, prices_daily, momentum_perturbed,
                sector_mapping, sector_limits, min_weight
            )
            if not perturbed_weights or len(perturbed_weights) < 12:
                failed_sims += 1
                continue

            # RUN BACKTEST
            equity_curve, strategy_rets = run_backtest_silent(
                portfolio_weights=perturbed_weights, monthly_returns=monthly_returns,
                enable_transaction_costs=enable_transaction_costs,
                transaction_cost_params=transaction_cost_params,
                enable_turnover_constraint=enable_turnover_constraint,
                max_turnover=max_turnover, min_trade_threshold=min_trade_threshold,
                min_position_weight=min_position_weight
            )
            if len(equity_curve) == 0:
                failed_sims += 1
                continue

            total_ret = (equity_curve.iloc[-1] / 100) - 1
            n_m = len(strategy_rets)
            ann_ret = (1 + total_ret) ** (12 / n_m) - 1
            sharpe = (strategy_rets.mean() / strategy_rets.std()) * (12 ** 0.5) if strategy_rets.std() > 0 else 0

            all_equity_curves.append(equity_curve)
            all_annual_returns.append(ann_ret)
            all_total_returns.append(total_ret)
            all_sharpe_ratios.append(sharpe)

        except Exception:
            failed_sims += 1
            continue

        if (sim + 1) % 50 == 0 or sim == 0:
            print(f" done (Ann: {ann_ret:.1%})")

    all_annual_returns = np.array(all_annual_returns)
    all_total_returns = np.array(all_total_returns)
    all_sharpe_ratios = np.array(all_sharpe_ratios)
    n_successful = len(all_annual_returns)

    if n_successful == 0:
        print(f"\n  All {n_simulations} simulations failed!")
        return None

    stats = {
        'mean_ann_return': np.mean(all_annual_returns),
        'median_ann_return': np.median(all_annual_returns),
        'std_ann_return': np.std(all_annual_returns),
        'min_ann_return': np.min(all_annual_returns),
        'max_ann_return': np.max(all_annual_returns),
        'p5_ann_return': np.percentile(all_annual_returns, 5),
        'p25_ann_return': np.percentile(all_annual_returns, 25),
        'p75_ann_return': np.percentile(all_annual_returns, 75),
        'p95_ann_return': np.percentile(all_annual_returns, 95),
        'mean_sharpe': np.mean(all_sharpe_ratios),
        'median_sharpe': np.median(all_sharpe_ratios),
        'std_sharpe': np.std(all_sharpe_ratios),
        'min_sharpe': np.min(all_sharpe_ratios),
        'p5_sharpe': np.percentile(all_sharpe_ratios, 5),
        'pct_positive_alpha': np.mean(all_annual_returns > 0) * 100,
        'pct_above_10pct': np.mean(all_annual_returns > 0.10) * 100,
        'pct_above_15pct': np.mean(all_annual_returns > 0.15) * 100,
        'pct_above_20pct': np.mean(all_annual_returns > 0.20) * 100,
        'pct_negative': np.mean(all_annual_returns < 0) * 100,
        'n_simulations': n_simulations, 'n_successful': n_successful,
        'n_failed': failed_sims, 'noise_scale': noise_scale,
    }

    return {
        'all_equity_curves': all_equity_curves,
        'all_annual_returns': all_annual_returns,
        'all_total_returns': all_total_returns,
        'all_sharpe_ratios': all_sharpe_ratios,
        'stats': stats,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _calc_max_drawdown_path(equity_path):
    peak = np.maximum.accumulate(equity_path)
    drawdown = (equity_path - peak) / peak
    return drawdown.min()

def _calc_skewness(returns):
    n = len(returns)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0: return 0
    return (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)

def _calc_kurtosis(returns):
    n = len(returns)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0: return 0
    return np.mean(((returns - mean) / std) ** 4) - 3


# =============================================================================
# APPROACH 1: REPORT
# =============================================================================

def print_monte_carlo_report(mc_results):
    stats = mc_results['stats']
    n_months = stats['n_months']
    years = n_months / 12

    print(f"\n{'=' * 70}")
    print(f"BOOTSTRAP MONTE CARLO RESULTS")
    print(f"{'=' * 70}")
    print(f"  Simulations: {stats['n_simulations']:,}")
    print(f"  Horizon: {n_months} months ({years:.1f} years)")
    print(f"  Historical data used: {stats['n_historical_months']} months")
    print(f"\n  PROJECTED RETURNS ({years:.0f}-Year Horizon):")
    print(f"     Mean annualized return:   {stats['mean_ann_return']:.2%}")
    print(f"     Median annualized return: {stats['median_ann_return']:.2%}")
    print(f"     Mean terminal value:      {stats['mean_terminal']:.2f}")
    print(f"     Median terminal value:    {stats['median_terminal']:.2f}")
    print(f"\n  OUTCOME PROBABILITIES:")
    print(f"     P(loss after {years:.0f}y):       {stats['prob_loss']:.1%}")
    print(f"     P(>20% loss after {years:.0f}y):  {stats['prob_below_80']:.1%}")
    print(f"     P(double after {years:.0f}y):     {stats['prob_double']:.1%}")
    print(f"\n  RISK METRICS:")
    print(f"     Value at Risk (5%):       {stats['var_5pct']:.2f}")
    print(f"     Value at Risk (1%):       {stats['var_1pct']:.2f}")
    print(f"     CVaR (5%):                {stats['cvar_5pct']:.2f}")
    print(f"     CVaR (1%):                {stats['cvar_1pct']:.2f}")
    print(f"\n  DRAWDOWN ANALYSIS:")
    print(f"     Median max drawdown:      {stats['median_max_drawdown']:.1%}")
    print(f"     Mean max drawdown:        {stats['mean_max_drawdown']:.1%}")
    print(f"     Worst 5% max drawdown:    {stats['worst_max_drawdown_95']:.1%}")
    print(f"\n  RANGE OF OUTCOMES (Base 100):")
    print(f"     95th percentile (bull):   {stats['best_case_95']:.2f}")
    print(f"     Median:                   {stats['median_terminal']:.2f}")
    print(f"     5th percentile (bear):    {stats['worst_case_5']:.2f}")
    print(f"{'=' * 70}")


# =============================================================================
# APPROACH 2: REPORT
# =============================================================================

def print_robustness_report(robustness_results, original_ann_return=None):
    stats = robustness_results['stats']

    print(f"\n{'=' * 70}")
    print(f"ROBUSTNESS MONTE CARLO RESULTS")
    print(f"{'=' * 70}")
    print(f"  Simulations: {stats['n_successful']}/{stats['n_simulations']} successful")
    print(f"  Noise scale: {stats['noise_scale']:.0%} (momentum perturbation)")

    if original_ann_return is not None:
        print(f"\n  ORIGINAL STRATEGY (unperturbed):")
        print(f"     Annual return: {original_ann_return:.2%}")

    print(f"\n  PERTURBED RETURN DISTRIBUTION:")
    print(f"     Mean annualized return:   {stats['mean_ann_return']:.2%}")
    print(f"     Median annualized return: {stats['median_ann_return']:.2%}")
    print(f"     Std deviation:            {stats['std_ann_return']:.2%}")
    print(f"     5th percentile:           {stats['p5_ann_return']:.2%}")
    print(f"     25th percentile:          {stats['p25_ann_return']:.2%}")
    print(f"     75th percentile:          {stats['p75_ann_return']:.2%}")
    print(f"     95th percentile:          {stats['p95_ann_return']:.2%}")
    print(f"     Worst case:               {stats['min_ann_return']:.2%}")
    print(f"     Best case:                {stats['max_ann_return']:.2%}")

    print(f"\n  SHARPE RATIO DISTRIBUTION:")
    print(f"     Mean Sharpe:   {stats['mean_sharpe']:.2f}")
    print(f"     Median Sharpe: {stats['median_sharpe']:.2f}")
    print(f"     Worst 5%:      {stats['p5_sharpe']:.2f}")

    print(f"\n  ROBUSTNESS SCORECARD:")
    print(f"     % sims with positive return: {stats['pct_positive_alpha']:.1f}%")
    print(f"     % sims with >10% annual:     {stats['pct_above_10pct']:.1f}%")
    print(f"     % sims with >15% annual:     {stats['pct_above_15pct']:.1f}%")
    print(f"     % sims with >20% annual:     {stats['pct_above_20pct']:.1f}%")
    print(f"     % sims with negative return: {stats['pct_negative']:.1f}%")

    if original_ann_return is not None:
        median_vs_original = stats['median_ann_return'] / original_ann_return if original_ann_return != 0 else 0
        print(f"\n  ROBUSTNESS VERDICT:")
        if stats['pct_positive_alpha'] >= 95 and median_vs_original >= 0.7:
            print(f"     HIGHLY ROBUST - Alpha survives {stats['noise_scale']:.0%} noise in {stats['pct_positive_alpha']:.0f}% of cases")
        elif stats['pct_positive_alpha'] >= 80 and median_vs_original >= 0.5:
            print(f"     MODERATELY ROBUST - Alpha holds in {stats['pct_positive_alpha']:.0f}% of cases but sensitive to stock selection")
        elif stats['pct_positive_alpha'] >= 60:
            print(f"     FRAGILE - Alpha drops significantly with perturbation ({stats['pct_positive_alpha']:.0f}% positive)")
        else:
            print(f"     NOT ROBUST - Alpha likely driven by specific stock picks ({stats['pct_positive_alpha']:.0f}% positive)")
        print(f"     Median perturbed / Original: {median_vs_original:.0%}")
        print(f"     Return volatility across sims: {stats['std_ann_return']:.2%}")

    print(f"{'=' * 70}")


# =============================================================================
# APPROACH 1: PLOT
# =============================================================================

def plot_monte_carlo(mc_results, title="Monte Carlo Simulation",
                     n_sample_paths=200, save_path=None):
    equity_paths = mc_results['equity_paths']
    percentile_paths = mc_results['percentile_paths']
    terminal_values = mc_results['terminal_values']
    drawdown_dist = mc_results['drawdown_distribution']
    monthly_stats = mc_results['monthly_stats']
    stats = mc_results['stats']
    n_months = stats['n_months']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    months = np.arange(n_months + 1)

    # Panel 1: Fan Chart
    ax1 = axes[0, 0]
    sample_idx = np.random.choice(len(equity_paths), min(n_sample_paths, len(equity_paths)), replace=False)
    for idx in sample_idx:
        ax1.plot(months, equity_paths[idx], color='#4A90D9', alpha=0.03, linewidth=0.5)
    if 'p5' in percentile_paths.columns and 'p95' in percentile_paths.columns:
        ax1.fill_between(months, percentile_paths['p5'], percentile_paths['p95'], alpha=0.15, color='#2E86AB', label='5th-95th pctile')
    if 'p25' in percentile_paths.columns and 'p75' in percentile_paths.columns:
        ax1.fill_between(months, percentile_paths['p25'], percentile_paths['p75'], alpha=0.25, color='#2E86AB', label='25th-75th pctile')
    if 'p10' in percentile_paths.columns and 'p90' in percentile_paths.columns:
        ax1.fill_between(months, percentile_paths['p10'], percentile_paths['p90'], alpha=0.18, color='#2E86AB', label='10th-90th pctile')
    if 'p50' in percentile_paths.columns:
        ax1.plot(months, percentile_paths['p50'], color='#E74C3C', linewidth=2.5, label='Median', zorder=5)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Projected Equity Paths', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Months Forward'); ax1.set_ylabel('Portfolio Value (Base 100)')
    ax1.legend(loc='upper left', fontsize=9); ax1.grid(True, alpha=0.3)

    # Panel 2: Terminal Value Distribution
    ax2 = axes[0, 1]
    n_bins = min(100, max(30, len(terminal_values) // 100))
    ax2.hist(terminal_values, bins=n_bins, density=True, alpha=0.7, color='#2E86AB', edgecolor='white', linewidth=0.3)
    ax2.axvline(x=stats['var_5pct'], color='#E74C3C', linestyle='--', linewidth=1.5, label=f'5th: {stats["var_5pct"]:.1f}')
    ax2.axvline(x=stats['median_terminal'], color='#F39C12', linestyle='-', linewidth=2, label=f'Median: {stats["median_terminal"]:.1f}')
    ax2.axvline(x=stats['best_case_95'], color='#27AE60', linestyle='--', linewidth=1.5, label=f'95th: {stats["best_case_95"]:.1f}')
    ax2.axvline(x=100, color='gray', linestyle=':', linewidth=1, label='Break-even')
    ax2.set_title(f'Terminal Value Distribution ({n_months//12}Y)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Terminal Portfolio Value'); ax2.set_ylabel('Density')
    ax2.legend(loc='upper right', fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Max Drawdown Distribution
    ax3 = axes[1, 0]
    dd_pct = drawdown_dist * 100
    ax3.hist(dd_pct, bins=60, density=True, alpha=0.7, color='#E74C3C', edgecolor='white', linewidth=0.3)
    ax3.axvline(x=np.median(dd_pct), color='#F39C12', linestyle='-', linewidth=2, label=f'Median: {np.median(dd_pct):.1f}%')
    ax3.axvline(x=np.percentile(dd_pct, 5), color='darkred', linestyle='--', linewidth=1.5, label=f'Worst 5%: {np.percentile(dd_pct, 5):.1f}%')
    ax3.set_title('Max Drawdown Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Maximum Drawdown (%)'); ax3.set_ylabel('Density')
    ax3.legend(loc='upper left', fontsize=9); ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Probability of Loss Over Time
    ax4 = axes[1, 1]
    prob_loss = monthly_stats['prob_loss'].values * 100
    ax4.plot(months, prob_loss, color='#E74C3C', linewidth=2.5)
    ax4.fill_between(months, 0, prob_loss, alpha=0.2, color='#E74C3C')
    ax4.set_title('Probability of Being Underwater', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Months Forward'); ax4.set_ylabel('Probability of Loss (%)')
    ax4.set_ylim(0, max(prob_loss.max() * 1.1, 10))
    ax4.grid(True, alpha=0.3); ax4.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Bootstrap MC plot saved as '{save_path}'")
    plt.show()


# =============================================================================
# APPROACH 2: PLOT
# =============================================================================

def plot_robustness_monte_carlo(robustness_results, original_equity_curve=None,
                                 original_ann_return=None, title="Robustness Monte Carlo",
                                 save_path=None):
    all_equity_curves = robustness_results['all_equity_curves']
    all_annual_returns = robustness_results['all_annual_returns']
    all_sharpe_ratios = robustness_results['all_sharpe_ratios']
    stats = robustness_results['stats']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    # Panel 1: All Perturbed Equity Curves
    ax1 = axes[0, 0]
    for eq_curve in all_equity_curves:
        ax1.plot(eq_curve.index, eq_curve.values, color='#4A90D9', alpha=0.08, linewidth=0.5)
    if original_equity_curve is not None:
        ax1.plot(original_equity_curve.index, original_equity_curve.values,
                 color='#E74C3C', linewidth=2.5, label='Original Strategy', zorder=10)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'Perturbed Equity Curves ({stats["n_successful"]} sims)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Portfolio Value (Base 100)')
    ax1.legend(loc='upper left', fontsize=10); ax1.grid(True, alpha=0.3)

    # Panel 2: Annualized Return Distribution
    ax2 = axes[0, 1]
    ann_ret_pct = all_annual_returns * 100
    ax2.hist(ann_ret_pct, bins=40, density=True, alpha=0.7, color='#2E86AB', edgecolor='white', linewidth=0.3)
    ax2.axvline(x=np.median(ann_ret_pct), color='#F39C12', linestyle='-', linewidth=2, label=f'Median: {np.median(ann_ret_pct):.1f}%')
    ax2.axvline(x=np.percentile(ann_ret_pct, 5), color='#E74C3C', linestyle='--', linewidth=1.5, label=f'5th: {np.percentile(ann_ret_pct, 5):.1f}%')
    ax2.axvline(x=np.percentile(ann_ret_pct, 95), color='#27AE60', linestyle='--', linewidth=1.5, label=f'95th: {np.percentile(ann_ret_pct, 95):.1f}%')
    if original_ann_return is not None:
        ax2.axvline(x=original_ann_return * 100, color='#E74C3C', linestyle='-', linewidth=2.5, label=f'Original: {original_ann_return*100:.1f}%', zorder=10)
    ax2.set_title('Annualized Return Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Annualized Return (%)'); ax2.set_ylabel('Density')
    ax2.legend(loc='best', fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Sharpe Ratio Distribution
    ax3 = axes[1, 0]
    ax3.hist(all_sharpe_ratios, bins=40, density=True, alpha=0.7, color='#27AE60', edgecolor='white', linewidth=0.3)
    ax3.axvline(x=np.median(all_sharpe_ratios), color='#F39C12', linestyle='-', linewidth=2, label=f'Median: {np.median(all_sharpe_ratios):.2f}')
    ax3.axvline(x=np.percentile(all_sharpe_ratios, 5), color='#E74C3C', linestyle='--', linewidth=1.5, label=f'5th: {np.percentile(all_sharpe_ratios, 5):.2f}')
    ax3.axvline(x=0, color='gray', linestyle=':', linewidth=1, label='Zero')
    ax3.set_title('Sharpe Ratio Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sharpe Ratio'); ax3.set_ylabel('Density')
    ax3.legend(loc='upper right', fontsize=9); ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Percentile Envelope vs Original
    ax4 = axes[1, 1]
    if len(all_equity_curves) > 10:
        all_dates = set(all_equity_curves[0].index)
        for ec in all_equity_curves[1:]:
            all_dates = all_dates.intersection(set(ec.index))
        common_dates = sorted(all_dates)
        if len(common_dates) > 0:
            aligned = pd.DataFrame(index=common_dates)
            for i, ec in enumerate(all_equity_curves):
                aligned[f's{i}'] = ec.reindex(common_dates)
            p5 = aligned.quantile(0.05, axis=1)
            p25 = aligned.quantile(0.25, axis=1)
            p50 = aligned.quantile(0.50, axis=1)
            p75 = aligned.quantile(0.75, axis=1)
            p95 = aligned.quantile(0.95, axis=1)
            ax4.fill_between(common_dates, p5, p95, alpha=0.15, color='#2E86AB', label='5th-95th')
            ax4.fill_between(common_dates, p25, p75, alpha=0.30, color='#2E86AB', label='25th-75th')
            ax4.plot(common_dates, p50, color='#2E86AB', linewidth=2, linestyle='--', label='Median perturbed')
            if original_equity_curve is not None:
                orig_aligned = original_equity_curve.reindex(common_dates)
                ax4.plot(common_dates, orig_aligned.values, color='#E74C3C', linewidth=2.5, label='Original', zorder=10)
            ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Original vs Perturbed Envelope', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date'); ax4.set_ylabel('Portfolio Value (Base 100)')
    ax4.legend(loc='upper left', fontsize=9); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Robustness plot saved as '{save_path}'")
    plt.show()


# =============================================================================
# APPROACH 1: EXCEL EXPORT
# =============================================================================

def export_monte_carlo_to_excel(mc_results, filename='monte_carlo_results.xlsx'):
    stats = mc_results['stats']
    percentile_paths = mc_results['percentile_paths']
    terminal_values = mc_results['terminal_values']
    drawdown_dist = mc_results['drawdown_distribution']
    monthly_stats = mc_results['monthly_stats']

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            s = []
            s.append(('Simulations', f"{stats['n_simulations']:,}"))
            s.append(('Horizon (Months)', stats['n_months']))
            s.append(('Historical Months', stats['n_historical_months']))
            s.append(('', ''))
            s.append(('Mean Ann Return', f"{stats['mean_ann_return']:.2%}"))
            s.append(('Median Ann Return', f"{stats['median_ann_return']:.2%}"))
            s.append(('Mean Terminal', f"{stats['mean_terminal']:.2f}"))
            s.append(('Median Terminal', f"{stats['median_terminal']:.2f}"))
            s.append(('', ''))
            s.append(('P(Loss)', f"{stats['prob_loss']:.2%}"))
            s.append(('P(>20% Loss)', f"{stats['prob_below_80']:.2%}"))
            s.append(('P(Double)', f"{stats['prob_double']:.2%}"))
            s.append(('', ''))
            s.append(('VaR 5%', f"{stats['var_5pct']:.2f}"))
            s.append(('VaR 1%', f"{stats['var_1pct']:.2f}"))
            s.append(('CVaR 5%', f"{stats['cvar_5pct']:.2f}"))
            s.append(('CVaR 1%', f"{stats['cvar_1pct']:.2f}"))
            s.append(('', ''))
            s.append(('Median Max DD', f"{stats['median_max_drawdown']:.2%}"))
            s.append(('Mean Max DD', f"{stats['mean_max_drawdown']:.2%}"))
            s.append(('Worst 5% Max DD', f"{stats['worst_max_drawdown_95']:.2%}"))
            s.append(('', ''))
            s.append(('95th pctile', f"{stats['best_case_95']:.2f}"))
            s.append(('Median', f"{stats['median_terminal']:.2f}"))
            s.append(('5th pctile', f"{stats['worst_case_5']:.2f}"))
            pd.DataFrame(s, columns=['Metric', 'Value']).to_excel(writer, sheet_name='MC_Summary', index=False)

            percentile_paths.copy().to_excel(writer, sheet_name='MC_Percentile_Paths')
            monthly_stats.to_excel(writer, sheet_name='MC_Monthly_Stats')

            hc, he = np.histogram(terminal_values, bins=50)
            pd.DataFrame({'Bin_Center': (he[:-1]+he[1:])/2, 'Count': hc, 'Freq': hc/len(terminal_values)}).to_excel(writer, sheet_name='MC_Terminal_Dist', index=False)

            dc, de = np.histogram(drawdown_dist*100, bins=50)
            pd.DataFrame({'DD_Center': (de[:-1]+de[1:])/2, 'Count': dc, 'Freq': dc/len(drawdown_dist)}).to_excel(writer, sheet_name='MC_DD_Dist', index=False)

        print(f"  Bootstrap MC exported to '{filename}'")
    except Exception as e:
        print(f"  Bootstrap MC export failed: {e}")


# =============================================================================
# APPROACH 2: EXCEL EXPORT
# =============================================================================

def export_robustness_to_excel(robustness_results, original_ann_return=None,
                                filename='robustness_monte_carlo_results.xlsx'):
    stats = robustness_results['stats']
    all_annual_returns = robustness_results['all_annual_returns']
    all_sharpe_ratios = robustness_results['all_sharpe_ratios']
    all_total_returns = robustness_results['all_total_returns']

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            s = []
            s.append(('Simulations', stats['n_simulations']))
            s.append(('Successful', stats['n_successful']))
            s.append(('Failed', stats['n_failed']))
            s.append(('Noise Scale', f"{stats['noise_scale']:.0%}"))
            s.append(('', ''))
            if original_ann_return is not None:
                s.append(('Original Ann Return', f"{original_ann_return:.2%}"))
                s.append(('', ''))
            s.append(('Mean Ann Return', f"{stats['mean_ann_return']:.2%}"))
            s.append(('Median Ann Return', f"{stats['median_ann_return']:.2%}"))
            s.append(('Std Dev', f"{stats['std_ann_return']:.2%}"))
            s.append(('5th pctile', f"{stats['p5_ann_return']:.2%}"))
            s.append(('95th pctile', f"{stats['p95_ann_return']:.2%}"))
            s.append(('', ''))
            s.append(('Mean Sharpe', f"{stats['mean_sharpe']:.2f}"))
            s.append(('Median Sharpe', f"{stats['median_sharpe']:.2f}"))
            s.append(('', ''))
            s.append(('% Positive', f"{stats['pct_positive_alpha']:.1f}%"))
            s.append(('% >10% Ann', f"{stats['pct_above_10pct']:.1f}%"))
            s.append(('% >15% Ann', f"{stats['pct_above_15pct']:.1f}%"))
            s.append(('% >20% Ann', f"{stats['pct_above_20pct']:.1f}%"))
            pd.DataFrame(s, columns=['Metric', 'Value']).to_excel(writer, sheet_name='Robustness_Summary', index=False)

            pd.DataFrame({
                'Sim': range(1, len(all_annual_returns)+1),
                'Ann_Return_Pct': (all_annual_returns*100).round(2),
                'Total_Return_Pct': (all_total_returns*100).round(2),
                'Sharpe': all_sharpe_ratios.round(3),
            }).to_excel(writer, sheet_name='Sim_Details', index=False)

            hc, he = np.histogram(all_annual_returns*100, bins=40)
            pd.DataFrame({'Return_Pct': ((he[:-1]+he[1:])/2).round(2), 'Count': hc}).to_excel(writer, sheet_name='Return_Dist', index=False)

        print(f"  Robustness MC exported to '{filename}'")
    except Exception as e:
        print(f"  Robustness MC export failed: {e}")
