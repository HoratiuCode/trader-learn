from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    closes: list[float]
    volumes: list[float]
    liquidities: list[float]
    wick: float


@dataclass(frozen=True)
class MemeCoinSpec:
    symbol: str
    scenario: Scenario


SCENARIOS = [
    Scenario(
        name="trend_breakout",
        closes=[1.00, 1.02, 1.04, 1.05, 1.08, 1.11, 1.15, 1.18, 1.16, 1.20],
        volumes=[1000, 1100, 1200, 1250, 1400, 1600, 1800, 2200, 2100, 2300],
        liquidities=[50000, 50500, 51000, 51500, 52000, 53000, 54000, 55000, 54800, 56000],
        wick=0.012,
    ),
    Scenario(
        name="fake_breakout",
        closes=[1.20, 1.23, 1.27, 1.31, 1.35, 1.29, 1.24, 1.20, 1.18, 1.16],
        volumes=[1500, 1600, 2000, 2400, 3000, 2900, 2500, 2000, 1700, 1600],
        liquidities=[52000, 51500, 51000, 50000, 49000, 48000, 47000, 46500, 46000, 45500],
        wick=0.02,
    ),
    Scenario(
        name="vertical_pump_dump",
        closes=[1.16, 1.25, 1.37, 1.52, 1.68, 1.82, 1.90, 1.78, 1.62, 1.45],
        volumes=[1800, 2600, 4200, 6500, 9000, 11000, 10000, 8000, 6000, 4000],
        liquidities=[47000, 47500, 48000, 48500, 49000, 49500, 49000, 47000, 45000, 43000],
        wick=0.03,
    ),
    Scenario(
        name="unstable_reversal",
        closes=[1.45, 1.38, 1.32, 1.25, 1.20, 1.24, 1.30, 1.27, 1.31, 1.34],
        volumes=[2200, 2400, 2600, 3000, 3400, 3600, 3300, 3100, 2900, 2800],
        liquidities=[43000, 42500, 42000, 41500, 41000, 40800, 41200, 41500, 42000, 42500],
        wick=0.022,
    ),
    Scenario(
        name="trend_pullback",
        closes=[1.34, 1.31, 1.29, 1.30, 1.34, 1.39, 1.45, 1.52, 1.58, 1.64],
        volumes=[1600, 1700, 1800, 1900, 2200, 2500, 2800, 3200, 3400, 3600],
        liquidities=[44000, 44200, 44500, 44800, 45200, 45800, 46500, 47200, 47800, 48500],
        wick=0.015,
    ),
    Scenario(
        name="chop",
        closes=[1.64, 1.63, 1.65, 1.62, 1.64, 1.61, 1.63, 1.62, 1.64, 1.63],
        volumes=[900, 950, 970, 1000, 1020, 980, 990, 1010, 1030, 990],
        liquidities=[46000, 46100, 46000, 46200, 46150, 46100, 46250, 46180, 46220, 46200],
        wick=0.01,
    ),
    Scenario(
        name="breakout_continuation",
        closes=[1.63, 1.68, 1.74, 1.82, 1.90, 1.86, 1.93, 2.00, 2.08, 2.16],
        volumes=[1400, 1700, 2200, 3000, 4200, 4000, 4500, 5000, 5200, 5400],
        liquidities=[47000, 47600, 48200, 48800, 49500, 50200, 51000, 51800, 52500, 53200],
        wick=0.018,
    ),
    Scenario(
        name="exhaustion_recovery",
        closes=[2.16, 2.10, 2.04, 1.98, 1.95, 2.00, 2.06, 2.13, 2.20, 2.28],
        volumes=[2000, 2300, 2400, 2600, 2800, 3000, 3300, 3600, 3900, 4200],
        liquidities=[52000, 51500, 51000, 50500, 50200, 50500, 51000, 51500, 52000, 52800],
        wick=0.02,
    ),
]


MEMECOINS = [
    MemeCoinSpec(symbol="PUPILO", scenario=SCENARIOS[0]),
    MemeCoinSpec(symbol="DECODE", scenario=SCENARIOS[1]),
    MemeCoinSpec(symbol="COFFE67", scenario=SCENARIOS[2]),
    MemeCoinSpec(symbol="NULLO", scenario=SCENARIOS[3]),
    MemeCoinSpec(symbol="VIBE7", scenario=SCENARIOS[4]),
    MemeCoinSpec(symbol="GLITCH", scenario=SCENARIOS[5]),
    MemeCoinSpec(symbol="BOUNCE9", scenario=SCENARIOS[6]),
    MemeCoinSpec(symbol="ZAZA", scenario=SCENARIOS[7]),
    MemeCoinSpec(symbol="MEME", scenario=SCENARIOS[6]),
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_timestamp = datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc)
    for coin_index, coin in enumerate(MEMECOINS):
        timestamp = base_timestamp + timedelta(days=coin_index)
        closes = _expand_pattern(coin.scenario.closes, 40, price_bias=0.035)
        volumes = _expand_pattern(coin.scenario.volumes, 40, price_bias=0.06)
        liquidities = _expand_pattern(coin.scenario.liquidities, 40, price_bias=0.04)
        previous_close = closes[0] * 0.98
        for close, volume, liquidity in zip(closes, volumes, liquidities):
            open_price = previous_close
            body = close - open_price
            direction = 1 if body >= 0 else -1
            wick = coin.scenario.wick
            high = max(open_price, close) * (1.0 + wick + abs(body) * 0.015)
            low = min(open_price, close) * (1.0 - wick - abs(body) * 0.010)
            if direction < 0:
                high = max(high, open_price * (1.0 + wick * 0.75))
            market_cap = liquidity * 1.85
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": coin.symbol,
                    "open": round(open_price, 6),
                    "high": round(high, 6),
                    "low": round(low, 6),
                    "close": round(close, 6),
                    "volume": round(volume, 2),
                    "liquidity": round(liquidity, 2),
                    "market_cap": round(market_cap, 2),
                    "scenario": coin.scenario.name,
                    "chart_type": coin.scenario.name,
                }
            )
            timestamp += timedelta(minutes=1)
            previous_close = close
    return rows


def _expand_pattern(values: list[float], target_length: int, *, price_bias: float) -> list[float]:
    if not values:
        return [0.0] * target_length
    if len(values) >= target_length:
        return values[:target_length]

    base_start = values[0]
    base_end = values[-1]
    direction = 1.0 if base_end >= base_start else -1.0
    cycle_count = ceil(target_length / len(values))
    expanded: list[float] = []
    for cycle in range(cycle_count):
        cycle_multiplier = 1.0 + direction * price_bias * cycle
        cycle_offset = direction * base_start * price_bias * 0.35 * cycle
        for value in values:
            adjusted = max(0.0001, value * cycle_multiplier + cycle_offset)
            expanded.append(round(adjusted, 6))
            if len(expanded) == target_length:
                return expanded
    return expanded[:target_length]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json_file(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = build_rows()
    write_csv(root / "data" / "sample_market_data.csv", rows)
    write_json_file(root / "data" / "sample_market_data.json", rows)
    sample_config = {
        "backtest": {
            "initial_capital": 10000.0,
            "risk_per_trade": 0.02,
            "fee_rate": 0.001,
            "slippage_rate": 0.0015,
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.08,
            "max_position_pct": 0.25,
            "max_holding_bars": 18,
            "min_confidence": 0.55,
            "exit_on_weak_signal": True,
        },
        "strategy": {
            "breakout_lookback": 5,
            "volume_spike_multiplier": 1.35,
            "liquidity_floor": 0.85,
            "trend_lookback": 5,
            "pullback_depth": 0.03,
            "reversal_body_ratio": 0.55,
            "volatility_floor": 0.008,
            "volatility_ceiling": 0.09,
            "exit_momentum_threshold": -0.004,
        },
    }
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "sample_config.json").write_text(json.dumps(sample_config, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} sample bars across {len(MEMECOINS)} memecoins to data/sample_market_data.csv and data/sample_market_data.json")


if __name__ == "__main__":
    main()
