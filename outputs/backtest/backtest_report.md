# Backtest Report - MemeCoinMomentumV1

## Summary
- Initial capital: $10,000.00
- Final equity: $11,555.99
- Total return: 15.56%
- Win rate: 66.67%
- Max drawdown: 3.16%
- Sharpe-like: 2.71
- Score: 79.63 / 100

## Score Breakdown
- profitability: 98.67
- consistency: 77.04
- risk_control: 100.00
- drawdown_control: 93.05
- timing: 55.96
- discipline: 0.00

## Trades
- #1 2026-04-20T00:03:00+00:00 -> 2026-04-20T00:06:00+00:00 net=$190.75 return=7.63% reason=take_profit
- #2 2026-04-20T00:06:00+00:00 -> 2026-04-20T00:11:00+00:00 net=$194.39 return=7.63% reason=take_profit
- #3 2026-04-20T00:14:00+00:00 -> 2026-04-20T00:15:00+00:00 net=$-112.68 return=-4.34% reason=stop_loss
- #4 2026-04-20T00:22:00+00:00 -> 2026-04-20T00:23:00+00:00 net=$195.95 return=7.63% reason=take_profit
- #5 2026-04-20T00:23:00+00:00 -> 2026-04-20T00:24:00+00:00 net=$199.69 return=7.63% reason=take_profit
- #6 2026-04-20T00:24:00+00:00 -> 2026-04-20T00:25:00+00:00 net=$203.50 return=7.63% reason=take_profit
- #7 2026-04-20T00:25:00+00:00 -> 2026-04-20T00:27:00+00:00 net=$-117.95 return=-4.34% reason=stop_loss
- #8 2026-04-20T00:40:00+00:00 -> 2026-04-20T00:42:00+00:00 net=$-116.67 return=-4.34% reason=stop_loss
- #9 2026-04-20T00:45:00+00:00 -> 2026-04-20T00:47:00+00:00 net=$202.90 return=7.63% reason=take_profit
- #10 2026-04-20T00:47:00+00:00 -> 2026-04-20T00:49:00+00:00 net=$206.78 return=7.63% reason=take_profit
- #11 2026-04-20T00:52:00+00:00 -> 2026-04-20T00:53:00+00:00 net=$-63.80 return=-2.31% reason=signal_exit
- #12 2026-04-20T00:54:00+00:00 -> 2026-04-20T00:55:00+00:00 net=$-63.73 return=-2.32% reason=signal_exit
- #13 2026-04-20T01:01:00+00:00 -> 2026-04-20T01:03:00+00:00 net=$208.29 return=7.63% reason=take_profit
- #14 2026-04-20T01:03:00+00:00 -> 2026-04-20T01:07:00+00:00 net=$212.26 return=7.63% reason=take_profit
- #15 2026-04-20T01:07:00+00:00 -> 2026-04-20T01:09:00+00:00 net=$216.31 return=7.63% reason=take_profit

## Failure Analysis
Primary weakness: strategy loses often after 3-candle vertical pumps (2 trades, 230.63 total loss).
- strategy loses often after 3-candle vertical pumps (2)
- loss cluster around mid-liquidity / low-volume entries (2)
- entries are late in trend continuation trades (1)

### Recommendations
- Add features for volume exhaustion after parabolic candles.
- Delay entries until a post-pump consolidation forms.
- Add earlier trend strength features so entries do not chase late moves.

## ML Analysis
- Model: LinearPatternBiasModel
### Top Features
- max_favorable_excursion_pct: weight=1.9019
- liquidity_change_5: weight=1.3278
- liquidity_ratio_5: weight=1.1920
- upper_wick_pct: weight=-1.1642
- body_pct: weight=1.1580
- lower_wick_pct: weight=-1.1515
- volume_ratio_5: weight=1.0154
- distance_to_rolling_max_5: weight=0.8762
- return_1: weight=0.8047
- volume_change_5: weight=0.7779
### Recommendations
- Add wick exhaustion features to identify failed pump candles.
