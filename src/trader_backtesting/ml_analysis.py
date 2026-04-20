from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .models import ModelArtifact, TradeRecord
from .pattern_analysis import analyze_failures
from .utils import clamp, mean_or_zero, safe_div, sigmoid, stdev_or_zero


TRAINING_FEATURES = [
    "return_1",
    "return_3",
    "return_5",
    "volatility_5",
    "volume_ratio_5",
    "volume_change_5",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "momentum_3",
    "distance_to_rolling_max_5",
    "distance_to_rolling_min_5",
    "liquidity_ratio_5",
    "liquidity_change_5",
    "range_pct",
    "trend_strength_5",
    "max_favorable_excursion_pct",
    "max_adverse_excursion_pct",
]


def _pattern_label(trade: TradeRecord) -> str:
    features = trade.entry_features
    volume_ratio = float(features.get("volume_ratio_5", 0.0))
    liquidity_ratio = float(features.get("liquidity_ratio_5", 0.0))
    volatility = float(features.get("volatility_5", 0.0))
    momentum_3 = float(features.get("momentum_3", 0.0))
    distance_to_high = float(features.get("distance_to_rolling_max_5", 0.0))
    return_3 = float(features.get("return_3", 0.0))
    trend_strength = float(features.get("trend_strength_5", 0.0))
    lower = trade.entry_reason.lower()

    if "breakout" in lower and liquidity_ratio < 0.9:
        return "fake_breakout"
    if momentum_3 > 0.08 or return_3 > 0.10:
        return "vertical_pump"
    if "reversal" in lower and volatility > 0.03:
        return "unstable_reversal"
    if volume_ratio < 0.95 and volatility < 0.01:
        return "chop"
    if "trend" in lower or trend_strength > 0.01:
        return "trend_continuation"
    if distance_to_high > 0.02:
        return "breakout_chase"
    if liquidity_ratio >= 1.05 and volume_ratio >= 1.2:
        return "confirmed_momentum"
    return "mixed_setup"


def _feature_value(trade: TradeRecord, feature: str) -> float:
    if feature in trade.entry_features:
        return float(trade.entry_features.get(feature, 0.0))
    return float(getattr(trade, feature, 0.0))


def train_pattern_model(trades: list[TradeRecord]) -> ModelArtifact:
    if len(trades) < 2:
        raise ValueError("Need at least two trades to train the model.")

    labeled: list[tuple[TradeRecord, int, str]] = []
    for trade in trades:
        label = 1 if trade.net_pnl > 0 else 0
        pattern = _pattern_label(trade)
        labeled.append((trade, label, pattern))

    feature_stats: dict[str, dict[str, float]] = {}
    feature_weights: dict[str, float] = {}
    rankings: list[dict[str, Any]] = []
    positive_count = sum(label for _, label, _ in labeled)
    negative_count = len(labeled) - positive_count
    bias = math.log((positive_count + 1.0) / (negative_count + 1.0))

    for feature in TRAINING_FEATURES:
        values = [_feature_value(trade, feature) for trade, _, _ in labeled]
        labels = [label for _, label, _ in labeled]
        wins = [value for value, label in zip(values, labels) if label == 1]
        losses = [value for value, label in zip(values, labels) if label == 0]
        mean_win = mean_or_zero(wins)
        mean_loss = mean_or_zero(losses)
        std_all = max(stdev_or_zero(values), 1e-9)
        weight = safe_div(mean_win - mean_loss, std_all, 0.0)
        feature_stats[feature] = {
            "mean": mean_or_zero(values),
            "std": std_all,
            "mean_win": mean_win,
            "mean_loss": mean_loss,
            "support": float(len(values)),
        }
        feature_weights[feature] = weight
        rankings.append(
            {
                "feature": feature,
                "weight": round(weight, 6),
                "direction": "positive" if weight > 0 else "negative",
                "importance": round(abs(weight), 6),
                "mean_win": round(mean_win, 6),
                "mean_loss": round(mean_loss, 6),
                "support": len(values),
            }
        )

    rankings.sort(key=lambda item: item["importance"], reverse=True)

    pattern_summary = _pattern_summary(labeled)
    recommendations = _recommendation_set(pattern_summary, rankings)
    return ModelArtifact(
        model_name="LinearPatternBiasModel",
        bias=bias,
        feature_stats=feature_stats,
        feature_weights=feature_weights,
        label_stats={
            "trades": float(len(labeled)),
            "wins": float(positive_count),
            "losses": float(negative_count),
            "win_rate": safe_div(positive_count, len(labeled), 0.0),
        },
        pattern_summary=pattern_summary,
        feature_rankings=rankings[:10],
        recommendations=recommendations,
    )


def predict_trade_probability(model: ModelArtifact, trade: TradeRecord) -> float:
    score = model.bias
    for feature, weight in model.feature_weights.items():
        stats = model.feature_stats.get(feature, {})
        mean = stats.get("mean", 0.0)
        std = max(stats.get("std", 1.0), 1e-9)
        value = _feature_value(trade, feature)
        normalized = safe_div(value - mean, std, 0.0)
        score += weight * normalized
    return sigmoid(score)


def _pattern_summary(labeled: list[tuple[TradeRecord, int, str]]) -> dict[str, Any]:
    by_pattern: dict[str, list[tuple[TradeRecord, int]]] = defaultdict(list)
    for trade, label, pattern in labeled:
        by_pattern[pattern].append((trade, label))
    summary: dict[str, Any] = {}
    for pattern, items in by_pattern.items():
        count = len(items)
        wins = sum(label for _, label in items)
        pnl = sum(trade.net_pnl for trade, _ in items)
        summary[pattern] = {
            "count": count,
            "win_rate": safe_div(wins, count, 0.0),
            "average_pnl": pnl / count if count else 0.0,
        }
    return summary


def _recommendation_set(pattern_summary: dict[str, Any], rankings: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    low_performers = sorted(pattern_summary.items(), key=lambda item: item[1]["win_rate"])
    if low_performers:
        worst_pattern, worst_stats = low_performers[0]
        if worst_pattern == "fake_breakout":
            recommendations.append("Collect more examples of fast pump-then-dump patterns.")
            recommendations.append("Add a stronger low-liquidity filter before breakout entries.")
        elif worst_pattern == "vertical_pump":
            recommendations.append("Add features for volume exhaustion.")
            recommendations.append("Separate parabolic pump days from normal trend days.")
        elif worst_pattern == "unstable_reversal":
            recommendations.append("Delay reversal entries until a stronger confirmation candle appears.")
            recommendations.append("Track rejection wicks and failed retests as explicit features.")
        elif worst_pattern == "chop":
            recommendations.append("Separate trend days from chop days.")
            recommendations.append("Exclude muted-volume conditions from the entry logic.")

    top_negative = [item for item in rankings if item["direction"] == "negative"][:3]
    top_positive = [item for item in rankings if item["direction"] == "positive"][:3]
    if any(item["feature"] in {"liquidity_ratio_5", "liquidity_change_5"} for item in top_negative):
        recommendations.append("Model liquidity change more aggressively before entering meme coin breakouts.")
    if any(item["feature"] in {"body_pct", "upper_wick_pct", "lower_wick_pct"} for item in top_negative):
        recommendations.append("Add wick exhaustion features to identify failed pump candles.")
    if any(item["feature"] in {"trend_strength_5", "distance_to_rolling_max_5"} for item in top_positive):
        recommendations.append("Trend structure is valuable. Keep adding clearer trend-day examples.")
    if not recommendations:
        recommendations.append("Collect more labeled trades to stabilize the pattern model.")
    return _dedupe(recommendations)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def build_ml_report(trades: list[TradeRecord]) -> tuple[ModelArtifact, FailureAnalysisReport]:
    model = train_pattern_model(trades)
    failures = analyze_failures(trades)
    return model, failures
