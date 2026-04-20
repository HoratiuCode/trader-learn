# Failure Analysis

Primary weakness: strategy loses often after 3-candle vertical pumps (2 trades, 230.63 total loss).

## Clusters
- strategy loses often after 3-candle vertical pumps: 2 trades, loss=$230.63
- loss cluster around mid-liquidity / low-volume entries: 2 trades, loss=$180.47
- entries are late in trend continuation trades: 1 trades, loss=$63.73

## Recommendations
- Add features for volume exhaustion after parabolic candles.
- Delay entries until a post-pump consolidation forms.
- Add earlier trend strength features so entries do not chase late moves.
