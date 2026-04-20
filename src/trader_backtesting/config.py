from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BacktestConfig, StrategyConfig
from .utils import read_json, write_json


DEFAULT_APP_CONFIG = {
    "backtest": BacktestConfig().to_dict(),
    "strategy": StrategyConfig().to_dict(),
}


def load_app_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_APP_CONFIG.copy()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    merged = DEFAULT_APP_CONFIG.copy()
    if "backtest" in payload:
        merged["backtest"] = {**merged["backtest"], **payload["backtest"]}
    if "strategy" in payload:
        merged["strategy"] = {**merged["strategy"], **payload["strategy"]}
    for key, value in payload.items():
        if key not in {"backtest", "strategy"}:
            merged[key] = value
    return merged


def backtest_config_from_dict(payload: dict[str, Any]) -> BacktestConfig:
    defaults = BacktestConfig()
    data = defaults.to_dict()
    data.update(payload)
    return BacktestConfig(**data)


def strategy_config_from_dict(payload: dict[str, Any]) -> StrategyConfig:
    defaults = StrategyConfig()
    data = defaults.to_dict()
    data.update(payload)
    return StrategyConfig(**data)


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    write_json(path, config)
