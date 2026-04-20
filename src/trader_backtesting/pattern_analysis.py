from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .models import FailureAnalysisReport, FailureCluster, TradeRecord
from .utils import mean_or_zero, safe_div


@dataclass(slots=True)
class _BucketStats:
    count: int = 0
    total_loss: float = 0.0
    total_return: float = 0.0
    sample_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sample_reasons is None:
            self.sample_reasons = []


def _bucket(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def _trade_signature(trade: TradeRecord) -> tuple[str, dict[str, Any], list[str]]:
    features = trade.entry_features
    volume_ratio = float(features.get("volume_ratio_5", 0.0))
    liquidity_ratio = float(features.get("liquidity_ratio_5", 0.0))
    volatility = float(features.get("volatility_5", 0.0))
    momentum_3 = float(features.get("momentum_3", 0.0))
    return_3 = float(features.get("return_3", 0.0))
    body_ratio = float(features.get("body_pct", 0.0))
    close_location = float(features.get("close_location", 0.0))
    distance_to_high = float(features.get("distance_to_rolling_max_5", 0.0))

    tags: list[str] = []
    characteristics: dict[str, Any] = {
        "volume_bucket": _bucket(volume_ratio, 0.9, 1.35),
        "liquidity_bucket": _bucket(liquidity_ratio, 0.85, 1.15),
        "volatility_bucket": _bucket(volatility, 0.008, 0.04),
        "momentum_bucket": _bucket(momentum_3, -0.01, 0.05),
        "distance_bucket": _bucket(distance_to_high, -0.02, 0.02),
        "body_bucket": _bucket(body_ratio, 0.35, 0.70),
        "close_location_bucket": _bucket(close_location, 0.35, 0.80),
        "entry_reason": trade.entry_reason,
    }

    lower_reason = trade.entry_reason.lower()
    if "breakout" in lower_reason and liquidity_ratio < 0.9 and volume_ratio >= 1.2:
        tags.append("low_liquidity_fake_breakout")
    if (return_3 > 0.10 or momentum_3 > 0.08) and volume_ratio >= 1.3:
        tags.append("three_candle_vertical_pump")
    if "reversal" in lower_reason and volatility > 0.03:
        tags.append("unstable_reversal")
    if volume_ratio < 0.95 and volatility < 0.01:
        tags.append("chop_entry")
    if "trend" in lower_reason and distance_to_high < 0:
        tags.append("late_trend_entry")
    if not tags:
        tags.append("generic_loss")

    key = "|".join(sorted(tags) + [f"liq={characteristics['liquidity_bucket']}", f"vol={characteristics['volume_bucket']}", f"var={characteristics['volatility_bucket']}"])
    return key, characteristics, tags


def analyze_failures(trades: list[TradeRecord]) -> FailureAnalysisReport:
    losing_trades = [trade for trade in trades if trade.net_pnl <= 0]
    if not losing_trades:
        return FailureAnalysisReport(
            summary="No losing trades found in the current sample.",
            clusters=[],
            recommendations=["Collect more losing examples to analyze weak spots."],
        )

    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    characteristics_map: dict[str, dict[str, Any]] = {}
    tag_map: dict[str, list[str]] = {}
    for trade in losing_trades:
        key, characteristics, tags = _trade_signature(trade)
        buckets[key].append(trade)
        characteristics_map[key] = characteristics
        tag_map[key] = tags

    clusters: list[FailureCluster] = []
    for key, cluster_trades in sorted(buckets.items(), key=lambda item: sum(trade.net_pnl for trade in item[1])):
        loss_total = abs(sum(trade.net_pnl for trade in cluster_trades if trade.net_pnl < 0))
        avg_loss = sum(trade.net_pnl for trade in cluster_trades) / len(cluster_trades)
        label = _cluster_label(tag_map[key], characteristics_map[key], cluster_trades)
        clusters.append(
            FailureCluster(
                key=key,
                label=label,
                count=len(cluster_trades),
                total_loss=round(loss_total, 4),
                average_loss=round(avg_loss, 4),
                win_rate_against=0.0,
                sample_reasons=[trade.entry_reason for trade in cluster_trades[:3]],
                characteristics=characteristics_map[key],
            )
        )

    summary = _build_summary(clusters)
    recommendations = _recommendations(clusters)
    return FailureAnalysisReport(summary=summary, clusters=clusters, recommendations=recommendations)


def _cluster_label(tags: list[str], characteristics: dict[str, Any], trades: list[TradeRecord]) -> str:
    if "low_liquidity_fake_breakout" in tags:
        return "strategy fails most during low-liquidity fake breakouts"
    if "three_candle_vertical_pump" in tags:
        return "strategy loses often after 3-candle vertical pumps"
    if "unstable_reversal" in tags:
        return "entries are too early during unstable reversals"
    if "chop_entry" in tags:
        return "strategy gets chopped up in low-energy conditions"
    if "late_trend_entry" in tags:
        return "entries are late in trend continuation trades"
    volume = characteristics.get("volume_bucket", "mid")
    liquidity = characteristics.get("liquidity_bucket", "mid")
    return f"loss cluster around {liquidity}-liquidity / {volume}-volume entries"


def _build_summary(clusters: list[FailureCluster]) -> str:
    if not clusters:
        return "No failure clusters available."
    top = clusters[0]
    return f"Primary weakness: {top.label} ({top.count} trades, {top.total_loss:.2f} total loss)."


def _recommendations(clusters: list[FailureCluster]) -> list[str]:
    recommendations: list[str] = []
    labels = [cluster.label for cluster in clusters]
    joined = " ".join(labels).lower()
    if "fake breakout" in joined:
        recommendations.append("Collect more examples of fast pump-then-dump patterns.")
        recommendations.append("Add a stronger low-liquidity filter before breakout entries.")
    if "vertical pumps" in joined:
        recommendations.append("Add features for volume exhaustion after parabolic candles.")
        recommendations.append("Delay entries until a post-pump consolidation forms.")
    if "unstable reversals" in joined:
        recommendations.append("Require confirmation before reversal entries.")
        recommendations.append("Track wick rejection and failed retests more explicitly.")
    if "chop" in joined:
        recommendations.append("Separate trend days from chop days before trading.")
        recommendations.append("Avoid signals when volatility and volume are both muted.")
    if "late" in joined:
        recommendations.append("Add earlier trend strength features so entries do not chase late moves.")
    if not recommendations:
        recommendations.append("Collect more labeled trades to refine failure clustering.")
    return recommendations
