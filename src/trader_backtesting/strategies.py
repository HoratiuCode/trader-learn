from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import MarketBar, StrategyConfig, StrategyDecision
from .utils import clamp, mean_or_zero, safe_div


@dataclass(slots=True)
class StrategyContext:
    index: int
    bars: list[MarketBar]
    features: list[dict[str, Any]]
    position_open: bool
    bars_held: int = 0


class BaseStrategy:
    name = "BaseStrategy"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def decide(self, context: StrategyContext) -> StrategyDecision:
        raise NotImplementedError


class MemeCoinStrategy(BaseStrategy):
    name = "MemeCoinMomentumV1"

    def decide(self, context: StrategyContext) -> StrategyDecision:
        bar = context.bars[context.index]
        feature = context.features[context.index]
        bars = context.bars
        cfg = self.config

        recent = bars[max(0, context.index - cfg.trend_lookback) : context.index]
        recent_closes = [item.close for item in recent]
        recent_highs = [item.high for item in recent]
        recent_lows = [item.low for item in recent]
        recent_volume = [item.volume for item in recent]
        recent_liquidity = [item.liquidity or 0.0 for item in recent if item.liquidity is not None]
        score = 0.0
        tags: list[str] = []
        reasons: list[str] = []

        volume_ratio = float(feature.get("volume_ratio_5", 0.0))
        liquidity_ratio = float(feature.get("liquidity_ratio_5", 0.0))
        volatility = float(feature.get("volatility_5", 0.0))
        distance_to_high = float(feature.get("distance_to_rolling_max_5", 0.0))
        trend_strength = float(feature.get("trend_strength_5", 0.0))
        momentum_3 = float(feature.get("momentum_3", 0.0))
        return_3 = float(feature.get("return_3", 0.0))
        body_ratio = float(feature.get("body_pct", 0.0))
        close_location = float(feature.get("close_location", 0.0))

        if recent_highs:
            prior_high = max(recent_highs)
        else:
            prior_high = bar.high

        if bar.close > prior_high * 1.002 and volume_ratio >= cfg.volume_spike_multiplier:
            score += 0.48
            tags.append("momentum_breakout")
            reasons.append("breakout above recent high on expanded volume")

        if recent_closes:
            sma = mean_or_zero(recent_closes)
            pullback_depth = safe_div(sma - min(recent_lows) if recent_lows else 0.0, sma, 0.0)
            if bar.close > sma and pullback_depth >= cfg.pullback_depth and bar.close >= bar.high * 0.985:
                score += 0.30
                tags.append("trend_continuation_after_pullback")
                reasons.append("trend continuation after controlled pullback")

        if context.index > 0:
            prev = bars[context.index - 1]
            prev_body = abs(prev.close - prev.open)
            curr_body = abs(bar.close - bar.open)
            prev_range = max(prev.high - prev.low, 1e-9)
            curr_range = max(bar.high - bar.low, 1e-9)
            prev_down = prev.close < prev.open and safe_div(prev_body, prev_range, 0.0) >= cfg.reversal_body_ratio
            current_up = bar.close > bar.open and safe_div(curr_body, curr_range, 0.0) >= cfg.reversal_body_ratio
            if prev_down and current_up and bar.close > prev.open and volume_ratio >= 1.1:
                score += 0.36
                tags.append("rapid_reversal")
                reasons.append("rapid reversal after a strong sell candle")

        if volume_ratio >= cfg.volume_spike_multiplier:
            score += 0.14
            tags.append("volume_spike")
            reasons.append("volume expansion")

        if liquidity_ratio >= cfg.liquidity_floor:
            score += 0.10
            tags.append("liquidity_support")
            reasons.append("liquidity is supportive")
        elif feature.get("liquidity", 0.0) > 0:
            score -= 0.10
            reasons.append("liquidity is thin")

        if volatility >= cfg.volatility_floor and volatility <= cfg.volatility_ceiling and close_location >= 0.65:
            score += 0.12
            tags.append("high_volatility_entry")
            reasons.append("accepting a volatile candle with strong close")

        if distance_to_high >= -0.01 and trend_strength > 0:
            score += 0.10
            tags.append("price_near_high")
            reasons.append("price is near a fresh high")

        if momentum_3 > 0 and return_3 > 0:
            score += 0.08
            tags.append("positive_momentum")

        if volume_ratio < 0.95 and volatility < cfg.volatility_floor:
            score -= 0.25
            tags.append("chop_filter")
            reasons.append("volume and volatility are too muted")

        if momentum_3 < cfg.exit_momentum_threshold and context.position_open:
            return StrategyDecision(
                action="sell",
                confidence=0.70,
                reason="momentum faded after the move",
                tags=tags or ["momentum_fade"],
                score=score,
            )

        if context.position_open:
            if context.bars_held >= cfg.trend_lookback and momentum_3 <= cfg.exit_momentum_threshold and close_location < 0.5:
                return StrategyDecision(
                    action="sell",
                    confidence=0.62,
                    reason="trend weakened and price lost structure",
                    tags=tags or ["weak_structure"],
                    score=score,
                )
            return StrategyDecision(action="hold", confidence=clamp(score), reason="position active", tags=tags, score=score)

        if score >= 0.55:
            reason = "; ".join(reasons[:3]) if reasons else "entry conditions aligned"
            return StrategyDecision(
                action="buy",
                confidence=clamp(score),
                reason=reason,
                tags=tags,
                score=score,
            )

        return StrategyDecision(
            action="hold",
            confidence=clamp(score),
            reason="no high-conviction setup",
            tags=tags,
            score=score,
        )
