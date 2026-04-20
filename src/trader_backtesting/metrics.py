from __future__ import annotations

from typing import Any

from .models import BacktestResult, MetricBundle, TradeRecord
from .utils import safe_div, stdev_or_zero


def calculate_max_drawdown(equity_curve: list[dict[str, Any]]) -> float:
    peak = None
    max_drawdown = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        if peak is None or equity > peak:
            peak = equity
        if peak and peak > 0:
            drawdown = safe_div(peak - equity, peak, 0.0)
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def calculate_bar_returns(equity_curve: list[dict[str, Any]]) -> list[float]:
    returns: list[float] = []
    previous = None
    for point in equity_curve:
        equity = float(point["equity"])
        if previous is not None and previous > 0:
            returns.append(safe_div(equity - previous, previous, 0.0))
        previous = equity
    return returns


def compute_metrics(result: BacktestResult) -> MetricBundle:
    trades = result.trades
    equity_curve = result.equity_curve
    total_return = safe_div(result.final_equity - result.initial_capital, result.initial_capital, 0.0)
    max_drawdown = calculate_max_drawdown(equity_curve)
    bar_returns = calculate_bar_returns(equity_curve)
    sharpe_like = 0.0
    if bar_returns:
        avg_return = sum(bar_returns) / len(bar_returns)
        std_return = stdev_or_zero(bar_returns)
        sharpe_like = safe_div(avg_return, std_return, 0.0) * (len(bar_returns) ** 0.5 if len(bar_returns) > 0 else 0.0)

    winners = [trade for trade in trades if trade.net_pnl > 0]
    losers = [trade for trade in trades if trade.net_pnl <= 0]
    win_rate = safe_div(len(winners), len(trades), 0.0)
    average_win = sum(trade.net_pnl for trade in winners) / len(winners) if winners else 0.0
    average_loss = sum(trade.net_pnl for trade in losers) / len(losers) if losers else 0.0
    gross_profit = sum(trade.net_pnl for trade in winners)
    gross_loss = abs(sum(trade.net_pnl for trade in losers))
    profit_factor = safe_div(gross_profit, gross_loss, gross_profit if gross_profit > 0 else 0.0)
    expectancy = sum(trade.net_pnl for trade in trades) / len(trades) if trades else 0.0
    avg_bars_held = sum(trade.bars_held for trade in trades) / len(trades) if trades else 0.0
    avg_mfe = sum(trade.max_favorable_excursion_pct for trade in trades) / len(trades) if trades else 0.0
    avg_mae = sum(trade.max_adverse_excursion_pct for trade in trades) / len(trades) if trades else 0.0
    trade_frequency = safe_div(len(trades), max(len(equity_curve), 1), 0.0)
    consecutive_losses = _max_consecutive_losses(trades)
    recovery_factor = safe_div(result.final_equity - result.initial_capital, max_drawdown * result.initial_capital, 0.0) if max_drawdown > 0 else total_return

    metrics = {
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe_like,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_win": average_win,
        "average_loss": average_loss,
        "avg_bars_held": avg_bars_held,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "trade_frequency": trade_frequency,
        "consecutive_losses": float(consecutive_losses),
        "recovery_factor": recovery_factor,
    }
    trade_metrics = {
        "total_trades": float(len(trades)),
        "winners": float(len(winners)),
        "losers": float(len(losers)),
    }
    return MetricBundle(metrics=metrics, trade_metrics=trade_metrics)


def _max_consecutive_losses(trades: list[TradeRecord]) -> int:
    longest = 0
    current = 0
    for trade in trades:
        if trade.net_pnl <= 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
