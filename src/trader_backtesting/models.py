from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MarketBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = "MEME"
    liquidity: float | None = None
    market_cap: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(slots=True)
class StrategyDecision:
    action: str
    confidence: float
    reason: str
    tags: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    entry_time: datetime
    entry_index: int
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_reason: str
    entry_features: dict[str, float]
    entry_tags: list[str]
    entry_fee: float
    bars_held: int = 0
    max_price: float = 0.0
    min_price: float = 0.0


@dataclass(slots=True)
class TradeRecord:
    trade_id: int
    symbol: str
    entry_time: str
    exit_time: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    fees: float
    slippage_cost: float
    bars_held: int
    exit_reason: str
    entry_reason: str
    entry_tags: list[str]
    entry_features: dict[str, float]
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.02
    fee_rate: float = 0.001
    slippage_rate: float = 0.0015
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.08
    max_position_pct: float = 0.25
    max_holding_bars: int = 18
    min_confidence: float = 0.55
    allow_reentry: bool = True
    exit_on_weak_signal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StrategyConfig:
    breakout_lookback: int = 5
    volume_spike_multiplier: float = 1.4
    liquidity_floor: float = 0.85
    trend_lookback: int = 5
    pullback_depth: float = 0.03
    reversal_body_ratio: float = 0.55
    volatility_floor: float = 0.008
    volatility_ceiling: float = 0.09
    exit_momentum_threshold: float = -0.004

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataSummary:
    count: int
    symbols: list[str]
    start: str
    end: str
    columns: list[str]
    missing_liquidity: int
    missing_market_cap: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BacktestResult:
    strategy_name: str
    config: dict[str, Any]
    data_summary: dict[str, Any]
    trades: list[TradeRecord]
    equity_curve: list[dict[str, Any]]
    final_equity: float
    initial_capital: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "config": self.config,
            "data_summary": self.data_summary,
            "trades": [trade.to_dict() for trade in self.trades],
            "equity_curve": self.equity_curve,
            "final_equity": self.final_equity,
            "initial_capital": self.initial_capital,
            "notes": self.notes,
        }


@dataclass(slots=True)
class MetricBundle:
    metrics: dict[str, float]
    trade_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "trade_metrics": self.trade_metrics}


@dataclass(slots=True)
class ScoreBreakdown:
    total_score: float
    components: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureCluster:
    key: str
    label: str
    count: int
    total_loss: float
    average_loss: float
    win_rate_against: float
    sample_reasons: list[str]
    characteristics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureAnalysisReport:
    summary: str
    clusters: list[FailureCluster]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "recommendations": self.recommendations,
        }


@dataclass(slots=True)
class ModelArtifact:
    model_name: str
    bias: float
    feature_stats: dict[str, dict[str, float]]
    feature_weights: dict[str, float]
    label_stats: dict[str, float]
    pattern_summary: dict[str, Any]
    feature_rankings: list[dict[str, Any]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
