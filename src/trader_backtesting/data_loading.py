from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .models import DataSummary, MarketBar
from .utils import parse_timestamp, safe_div


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _coerce_row(row: dict[str, Any]) -> MarketBar:
    timestamp = parse_timestamp(row.get("timestamp") or row.get("time") or row.get("date"))
    symbol = str(row.get("symbol") or row.get("token") or "MEME").strip() or "MEME"
    liquidity = _parse_float(row.get("liquidity"))
    market_cap = _parse_float(row.get("market_cap") or row.get("marketcap"))
    return MarketBar(
        timestamp=timestamp,
        open=float(_parse_float(row.get("open"), 0.0) or 0.0),
        high=float(_parse_float(row.get("high"), 0.0) or 0.0),
        low=float(_parse_float(row.get("low"), 0.0) or 0.0),
        close=float(_parse_float(row.get("close"), 0.0) or 0.0),
        volume=float(_parse_float(row.get("volume"), 0.0) or 0.0),
        symbol=symbol,
        liquidity=liquidity,
        market_cap=market_cap,
        raw=dict(row),
    )


def load_market_data(path: str | Path) -> list[MarketBar]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Market data file not found: {source}")

    suffix = source.suffix.lower()
    rows: list[dict[str, Any]]
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [dict(item) for item in payload]
        elif isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                rows = [dict(item) for item in payload["data"]]
            else:
                rows = [dict(payload)]
        else:
            raise ValueError("Unsupported JSON market data shape.")
    else:
        raise ValueError(f"Unsupported market data extension: {suffix}")

    bars = [_coerce_row(row) for row in rows]
    bars.sort(key=lambda bar: (bar.symbol, bar.timestamp))
    return bars


def summarize_market_data(bars: list[MarketBar]) -> DataSummary:
    if not bars:
        raise ValueError("No market data rows found.")
    symbols = sorted({bar.symbol for bar in bars})
    columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "liquidity",
        "market_cap",
    ]
    missing_liquidity = sum(1 for bar in bars if bar.liquidity is None)
    missing_market_cap = sum(1 for bar in bars if bar.market_cap is None)
    return DataSummary(
        count=len(bars),
        symbols=symbols,
        start=bars[0].timestamp.isoformat(),
        end=bars[-1].timestamp.isoformat(),
        columns=columns,
        missing_liquidity=missing_liquidity,
        missing_market_cap=missing_market_cap,
    )


def save_normalized_market_data(path: str | Path, bars: list[MarketBar]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [bar.to_dict() for bar in bars]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_by_symbol(bars: list[MarketBar]) -> dict[str, list[MarketBar]]:
    grouped: dict[str, list[MarketBar]] = {}
    for bar in bars:
        grouped.setdefault(bar.symbol, []).append(bar)
    for symbol in grouped:
        grouped[symbol].sort(key=lambda bar: bar.timestamp)
    return grouped


def infer_basic_liquidity_ratio(bars: list[MarketBar]) -> list[float]:
    values = [bar.liquidity or 0.0 for bar in bars]
    if not values:
        return []
    maximum = max(values)
    if maximum <= 0:
        return [0.0 for _ in values]
    return [safe_div(value, maximum, 0.0) for value in values]
