# ML Analysis

- Model: LinearPatternBiasModel

## Feature Importance
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

## Pattern Summary
- trend_continuation: win_rate=0.40, count=5
- vertical_pump: win_rate=0.80, count=10

## Recommendations
- Add wick exhaustion features to identify failed pump candles.
