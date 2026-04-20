from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .features import FeatureEngineer
from .models import BacktestConfig, BacktestResult, MarketBar, OpenPosition, StrategyDecision, TradeRecord
from .strategies import BaseStrategy, StrategyContext
from .utils import safe_div


@dataclass(slots=True)
class SimulationState:
    cash: float
    position: OpenPosition | None = None
    trade_id: int = 1


def _apply_slippage(price: float, is_buy: bool, slippage_rate: float) -> float:
    return price * (1.0 + slippage_rate if is_buy else 1.0 - slippage_rate)


def _entry_quantity(cash: float, equity: float, price: float, stop_loss_pct: float, risk_per_trade: float, max_position_pct: float, confidence: float) -> float:
    stop_distance = max(price * stop_loss_pct, 1e-9)
    risk_budget = equity * risk_per_trade * max(confidence, 0.25)
    qty_by_risk = safe_div(risk_budget, stop_distance, 0.0)
    qty_by_capital = safe_div(equity * max_position_pct, price, 0.0)
    qty_by_cash = safe_div(cash, price, 0.0)
    quantity = min(qty_by_risk, qty_by_capital, qty_by_cash)
    return max(quantity, 0.0)


class Backtester:
    def __init__(self, config: BacktestConfig, strategy: BaseStrategy) -> None:
        self.config = config
        self.strategy = strategy

    def run(self, bars: list[MarketBar]) -> BacktestResult:
        if not bars:
            raise ValueError("No bars supplied to backtester.")

        feature_engineer = FeatureEngineer()
        feature_rows = feature_engineer.build(bars)
        state = SimulationState(cash=self.config.initial_capital)
        trades: list[TradeRecord] = []
        equity_curve: list[dict[str, Any]] = []
        notes: list[str] = []

        for index, bar in enumerate(bars):
            features = feature_rows[index]

            if state.position is not None:
                position = state.position
                position.bars_held += 1
                position.max_price = max(position.max_price, bar.high)
                if position.min_price == 0.0:
                    position.min_price = bar.low
                else:
                    position.min_price = min(position.min_price, bar.low)

                exit_reason: str | None = None
                exit_price: float | None = None

                if bar.low <= position.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
                elif bar.high >= position.take_profit:
                    exit_reason = "take_profit"
                    exit_price = position.take_profit
                elif position.bars_held >= self.config.max_holding_bars:
                    exit_reason = "time_exit"
                    exit_price = bar.close
                elif self.config.exit_on_weak_signal:
                    decision = self.strategy.decide(
                        StrategyContext(
                            index=index,
                            bars=bars,
                            features=feature_rows,
                            position_open=True,
                            bars_held=position.bars_held,
                        )
                    )
                    if decision.action == "sell":
                        exit_reason = "signal_exit"
                        exit_price = bar.close

                if exit_reason and exit_price is not None:
                    exit_fill = _apply_slippage(exit_price, is_buy=False, slippage_rate=self.config.slippage_rate)
                    gross_proceeds = position.quantity * exit_fill
                    exit_fee = gross_proceeds * self.config.fee_rate
                    entry_notional = position.quantity * position.entry_price
                    gross_pnl = gross_proceeds - entry_notional
                    net_pnl = gross_pnl - position.entry_fee - exit_fee
                    self._close_position(
                        state=state,
                        trades=trades,
                        trade_id=state.trade_id,
                        position=position,
                        exit_fill=exit_fill,
                        exit_time=bar.timestamp,
                        exit_index=index,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        exit_fee=exit_fee,
                        exit_reason=exit_reason,
                    )
                    state.trade_id += 1
                    state.position = None

            equity = state.cash
            if state.position is not None:
                equity += state.position.quantity * bar.close
            equity_curve.append(
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "equity": equity,
                    "cash": state.cash,
                    "position_value": 0.0 if state.position is None else state.position.quantity * bar.close,
                    "symbol": bar.symbol,
                }
            )

            if state.position is None:
                decision = self.strategy.decide(
                    StrategyContext(
                        index=index,
                        bars=bars,
                        features=feature_rows,
                        position_open=False,
                    )
                )
                if decision.action == "buy" and decision.confidence >= self.config.min_confidence:
                    entry_fill = _apply_slippage(bar.close, is_buy=True, slippage_rate=self.config.slippage_rate)
                    quantity = _entry_quantity(
                        cash=state.cash,
                        equity=equity,
                        price=entry_fill,
                        stop_loss_pct=self.config.stop_loss_pct,
                        risk_per_trade=self.config.risk_per_trade,
                        max_position_pct=self.config.max_position_pct,
                        confidence=decision.confidence,
                    )
                    if quantity > 0:
                        entry_notional = quantity * entry_fill
                        entry_fee = entry_notional * self.config.fee_rate
                        total_cost = entry_notional + entry_fee
                        if total_cost <= state.cash:
                            state.cash -= total_cost
                            state.position = OpenPosition(
                                symbol=bar.symbol,
                                entry_time=bar.timestamp,
                                entry_index=index,
                                entry_price=entry_fill,
                                quantity=quantity,
                                stop_loss=entry_fill * (1.0 - self.config.stop_loss_pct),
                                take_profit=entry_fill * (1.0 + self.config.take_profit_pct),
                                entry_reason=decision.reason,
                                entry_features={
                                    key: float(value)
                                    for key, value in features.items()
                                    if isinstance(value, (int, float))
                                },
                                entry_tags=list(decision.tags),
                                entry_fee=entry_fee,
                                max_price=bar.high,
                                min_price=bar.low,
                            )
                        else:
                            notes.append(f"Skipped entry at {bar.timestamp.isoformat()} due to insufficient cash.")

        if state.position is not None:
            last_bar = bars[-1]
            position = state.position
            exit_fill = _apply_slippage(last_bar.close, is_buy=False, slippage_rate=self.config.slippage_rate)
            gross_proceeds = position.quantity * exit_fill
            exit_fee = gross_proceeds * self.config.fee_rate
            entry_notional = position.quantity * position.entry_price
            gross_pnl = gross_proceeds - entry_notional
            net_pnl = gross_pnl - position.entry_fee - exit_fee
            self._close_position(
                state=state,
                trades=trades,
                trade_id=state.trade_id,
                position=position,
                exit_fill=exit_fill,
                exit_time=last_bar.timestamp,
                exit_index=len(bars) - 1,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                exit_fee=exit_fee,
                exit_reason="end_of_data",
            )
            state.position = None
            state.trade_id += 1
            equity_curve[-1]["equity"] = state.cash
            equity_curve[-1]["position_value"] = 0.0

        return BacktestResult(
            strategy_name=self.strategy.name,
            config=self.config.to_dict(),
            data_summary={
                "bars": len(bars),
                "symbol": bars[0].symbol,
                "start": bars[0].timestamp.isoformat(),
                "end": bars[-1].timestamp.isoformat(),
            },
            trades=trades,
            equity_curve=equity_curve,
            final_equity=state.cash,
            initial_capital=self.config.initial_capital,
            notes=notes,
        )

    def _close_position(
        self,
        state: SimulationState,
        trades: list[TradeRecord],
        trade_id: int,
        position: OpenPosition,
        exit_fill: float,
        exit_time: datetime,
        exit_index: int,
        gross_pnl: float,
        net_pnl: float,
        exit_fee: float,
        exit_reason: str,
    ) -> None:
        exit_notional = position.quantity * exit_fill
        state.cash += exit_notional - exit_fee
        total_fees = position.entry_fee + exit_fee
        return_pct = safe_div(net_pnl, position.quantity * position.entry_price, 0.0)
        mfe_pct = safe_div(position.max_price - position.entry_price, position.entry_price, 0.0)
        mae_pct = safe_div(position.entry_price - position.min_price, position.entry_price, 0.0)
        trades.append(
            TradeRecord(
                trade_id=trade_id,
                symbol=position.symbol,
                entry_time=position.entry_time.isoformat(),
                exit_time=exit_time.isoformat(),
                entry_index=position.entry_index,
                exit_index=exit_index,
                entry_price=position.entry_price,
                exit_price=exit_fill,
                quantity=position.quantity,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                return_pct=return_pct,
                fees=total_fees,
                slippage_cost=abs(position.entry_price - exit_fill) * position.quantity,
                bars_held=position.bars_held,
                exit_reason=exit_reason,
                entry_reason=position.entry_reason,
                entry_tags=position.entry_tags,
                entry_features=position.entry_features,
                max_favorable_excursion_pct=mfe_pct,
                max_adverse_excursion_pct=mae_pct,
            )
        )
