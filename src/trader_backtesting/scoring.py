from __future__ import annotations

from .models import MetricBundle, ScoreBreakdown, TradeRecord
from .utils import clamp, safe_div


def score_strategy(metrics: MetricBundle, trades: list[TradeRecord], bar_count: int) -> ScoreBreakdown:
    m = metrics.metrics
    total_return = float(m.get("total_return", 0.0))
    win_rate = float(m.get("win_rate", 0.0))
    max_drawdown = float(m.get("max_drawdown", 0.0))
    sharpe_like = float(m.get("sharpe_like", 0.0))
    profit_factor = float(m.get("profit_factor", 0.0))
    expectancy = float(m.get("expectancy", 0.0))
    avg_mfe = float(m.get("avg_mfe", 0.0))
    avg_mae = float(m.get("avg_mae", 0.0))
    avg_bars_held = float(m.get("avg_bars_held", 0.0))
    trade_frequency = safe_div(len(trades), max(bar_count, 1), 0.0)

    profitability = clamp(50.0 + total_return * 120.0 + min(profit_factor, 3.0) * 10.0, 0.0, 100.0)
    consistency = clamp(20.0 + win_rate * 70.0 + max(expectancy, 0.0) / max(float(m.get("initial_capital", 1.0)) * 0.01, 1.0) * 10.0, 0.0, 100.0)
    risk_control = clamp(100.0 - max_drawdown * 180.0 + sharpe_like * 8.0, 0.0, 100.0)
    drawdown_control = clamp(100.0 - max_drawdown * 220.0, 0.0, 100.0)
    timing = clamp(50.0 + (avg_mfe - avg_mae) * 250.0 + min(max(avg_bars_held, 1.0), 20.0), 0.0, 100.0)
    discipline = clamp(100.0 - trade_frequency * 900.0 - max(0.0, 3.0 - avg_bars_held) * 8.0, 0.0, 100.0)

    components = {
        "profitability": profitability,
        "consistency": consistency,
        "risk_control": risk_control,
        "drawdown_control": drawdown_control,
        "timing": timing,
        "discipline": discipline,
    }
    total_score = (
        profitability * 0.25
        + consistency * 0.20
        + risk_control * 0.20
        + drawdown_control * 0.15
        + timing * 0.10
        + discipline * 0.10
    )

    notes = []
    if total_return > 0:
        notes.append("Strategy is profitable on the sample period.")
    else:
        notes.append("Strategy is not profitable on the sample period.")
    if max_drawdown > 0.15:
        notes.append("Drawdown is elevated.")
    if trade_frequency > 0.15:
        notes.append("Trading frequency is high relative to the sample size.")
    if win_rate < 0.45:
        notes.append("Win rate is below a disciplined threshold.")
    if avg_mae > avg_mfe:
        notes.append("Adverse excursion is larger than favorable excursion on average.")

    return ScoreBreakdown(total_score=round(total_score, 2), components={k: round(v, 2) for k, v in components.items()}, notes=notes)
