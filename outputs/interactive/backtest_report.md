# Backtest Report - InteractiveSession:BOUNCE9

## Summary
- Initial capital: $1,000.00
- Final equity: $997.50
- Total return: -0.25%
- Win rate: 0.00%
- Max drawdown: 0.25%
- Sharpe-like: -56959.24
- Score: 35.88 / 100

## Score Breakdown
- profitability: 49.70
- consistency: 20.00
- risk_control: 0.00
- drawdown_control: 99.45
- timing: 45.39
- discipline: 0.00

## Trades
- #1 2026-04-26T00:00:00+00:00 -> 2026-04-26T00:00:00+00:00 net=$-2.50 return=-0.50% reason=end_of_session

## Failure Analysis
Primary weakness: strategy gets chopped up in low-energy conditions (1 trades, 2.50 total loss).
- strategy gets chopped up in low-energy conditions (1)

### Recommendations
- Separate trend days from chop days before trading.
- Avoid signals when volatility and volume are both muted.
