from __future__ import annotations

from collections import deque
from typing import Any

from .models import MarketBar
from .utils import mean_or_zero, safe_div, stdev_or_zero


class FeatureEngineer:
    def __init__(self, windows: tuple[int, ...] = (1, 3, 5, 10)) -> None:
        self.windows = windows

    def build(self, bars: list[MarketBar]) -> list[dict[str, Any]]:
        if not bars:
            return []

        closes: deque[float] = deque(maxlen=max(self.windows) + 1)
        volumes: deque[float] = deque(maxlen=max(self.windows) + 1)
        liquidities: deque[float] = deque(maxlen=max(self.windows) + 1)
        outputs: list[dict[str, Any]] = []

        for index, bar in enumerate(bars):
            closes.append(bar.close)
            volumes.append(bar.volume)
            liquidities.append(bar.liquidity if bar.liquidity is not None else 0.0)

            features: dict[str, Any] = {
                "index": float(index),
                "timestamp": bar.timestamp.isoformat(),
                "symbol": bar.symbol,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "liquidity": bar.liquidity if bar.liquidity is not None else 0.0,
                "market_cap": bar.market_cap if bar.market_cap is not None else 0.0,
            }

            range_width = max(bar.high - bar.low, 1e-9)
            body = abs(bar.close - bar.open)
            candle_direction = 1.0 if bar.close >= bar.open else -1.0
            features["range_pct"] = safe_div(range_width, bar.close, 0.0)
            features["body_pct"] = safe_div(body, range_width, 0.0)
            features["upper_wick_pct"] = safe_div(bar.high - max(bar.open, bar.close), range_width, 0.0)
            features["lower_wick_pct"] = safe_div(min(bar.open, bar.close) - bar.low, range_width, 0.0)
            features["candle_direction"] = candle_direction
            features["close_location"] = safe_div(bar.close - bar.low, range_width, 0.0)

            for window in self.windows:
                if len(closes) > window:
                    reference_close = list(closes)[-window - 1]
                    window_closes = list(closes)[-(window + 1) : -1]
                    window_volumes = list(volumes)[-(window + 1) : -1]
                    window_liqs = list(liquidities)[-(window + 1) : -1]
                else:
                    reference_close = None
                    window_closes = list(closes)[:-1]
                    window_volumes = list(volumes)[:-1]
                    window_liqs = list(liquidities)[:-1]

                if reference_close is not None:
                    features[f"return_{window}"] = safe_div(bar.close - reference_close, reference_close, 0.0)
                    features[f"momentum_{window}"] = bar.close - reference_close
                else:
                    features[f"return_{window}"] = 0.0
                    features[f"momentum_{window}"] = 0.0

                if window_closes:
                    rolling_high = max(window_closes)
                    rolling_low = min(window_closes)
                    rolling_mean = mean_or_zero(window_closes)
                    rolling_volatility = stdev_or_zero(
                        [
                            safe_div(window_closes[i] - window_closes[i - 1], window_closes[i - 1], 0.0)
                            for i in range(1, len(window_closes))
                        ]
                    )
                    features[f"distance_to_rolling_max_{window}"] = safe_div(bar.close - rolling_high, rolling_high, 0.0)
                    features[f"distance_to_rolling_min_{window}"] = safe_div(bar.close - rolling_low, rolling_low, 0.0)
                    features[f"trend_strength_{window}"] = safe_div(bar.close - rolling_mean, rolling_mean, 0.0)
                    features[f"volatility_{window}"] = rolling_volatility
                else:
                    features[f"distance_to_rolling_max_{window}"] = 0.0
                    features[f"distance_to_rolling_min_{window}"] = 0.0
                    features[f"trend_strength_{window}"] = 0.0
                    features[f"volatility_{window}"] = 0.0

                if window_volumes:
                    volume_mean = mean_or_zero(window_volumes)
                    features[f"volume_ratio_{window}"] = safe_div(bar.volume, volume_mean, 0.0)
                    if len(window_volumes) > 1:
                        vol_changes = [
                            safe_div(window_volumes[i] - window_volumes[i - 1], window_volumes[i - 1], 0.0)
                            for i in range(1, len(window_volumes))
                        ]
                        features[f"volume_change_{window}"] = mean_or_zero(vol_changes)
                    else:
                        features[f"volume_change_{window}"] = 0.0
                else:
                    features[f"volume_ratio_{window}"] = 0.0
                    features[f"volume_change_{window}"] = 0.0

                if window_liqs and any(window_liqs):
                    liq_mean = mean_or_zero(window_liqs)
                    features[f"liquidity_ratio_{window}"] = safe_div(
                        bar.liquidity if bar.liquidity is not None else 0.0,
                        liq_mean,
                        0.0,
                    )
                    if len(window_liqs) > 1:
                        liq_changes = [
                            safe_div(window_liqs[i] - window_liqs[i - 1], window_liqs[i - 1], 0.0)
                            for i in range(1, len(window_liqs))
                        ]
                        features[f"liquidity_change_{window}"] = mean_or_zero(liq_changes)
                    else:
                        features[f"liquidity_change_{window}"] = 0.0
                else:
                    features[f"liquidity_ratio_{window}"] = 0.0
                    features[f"liquidity_change_{window}"] = 0.0

            outputs.append(features)
        return outputs
