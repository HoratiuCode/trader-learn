from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except Exception:  # pragma: no cover - rich is expected, but keep a fallback
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]

from .config import backtest_config_from_dict, load_app_config, strategy_config_from_dict
from .data_loading import load_market_data, split_by_symbol, summarize_market_data
from .features import FeatureEngineer
from .metrics import compute_metrics
from .ml_analysis import train_pattern_model
from .models import BacktestConfig, BacktestResult, MarketBar, OpenPosition, TradeRecord
from .pattern_analysis import analyze_failures
from .reporting import render_backtest_console, save_backtest_artifacts
from .scoring import score_strategy
from .strategies import MemeCoinStrategy
from .utils import clamp, ensure_dir, format_currency, format_pct, safe_div


@dataclass(slots=True)
class InteractiveSessionState:
    cash: float
    trade_id: int = 1
    position: OpenPosition | None = None
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class InteractiveTraderSession:
    def __init__(
        self,
        bars: list[MarketBar],
        *,
        output_dir: str | Path,
        config: BacktestConfig,
        window_size: int = 20,
        symbol: str | None = None,
        budget: float | None = None,
    ) -> None:
        if not bars:
            raise ValueError("No market data supplied for the interactive session.")

        self.console = Console() if Console is not None else None
        self.output_dir = ensure_dir(output_dir)
        self.backtest_config = config
        self.window_size = max(5, window_size)
        self.symbol = symbol
        self.initial_budget = budget
        self.all_bars = bars
        self.grouped = split_by_symbol(bars)
        self.selected_bars: list[MarketBar] = []
        self.feature_rows: list[dict[str, Any]] = []
        self.state: InteractiveSessionState | None = None
        self.current_index = 0
        self._skip_auto_manage_once = False

    def run(self) -> None:
        self._show_header()
        symbol = self.symbol or self._prompt_symbol()
        self.selected_bars = self.grouped[symbol]
        self.symbol = symbol
        if self.initial_budget is None:
            self.initial_budget = self._prompt_budget()
        self.feature_rows = FeatureEngineer().build(self.selected_bars)
        self.state = InteractiveSessionState(cash=float(self.initial_budget))
        self.current_index = 0
        self._append_equity_point("session_start")
        self._render_screen("Session ready. Use buy, sell, next, chart, status, help, or quit.")
        while self.current_index < len(self.selected_bars):
            bar = self.selected_bars[self.current_index]
            if self._skip_auto_manage_once:
                self._skip_auto_manage_once = False
            elif self._auto_manage_position(bar):
                self._append_equity_point("auto_exit")
            command = self._prompt_command()
            if not self._handle_command(command):
                break
        self._finalize_session()

    def _console(self) -> Console | None:
        return self.console

    def _show_header(self) -> None:
        console = self._console()
        if console is None:
            print("Memelearn interactive session")
            return
        console.rule("[bold cyan]Memelearn Interactive Start")
        console.print("Select a memecoin, set a budget, inspect the chart, and trade with commands.")

    def _prompt_symbol(self) -> str:
        console = self._console()
        symbols = sorted(self.grouped)
        if console:
            console.print("[bold]Available memecoins:[/bold]")
            for index, symbol in enumerate(symbols, start=1):
                marker = " (default)" if index == 1 else ""
                console.print(f"  {index}. {symbol}{marker}")
            while True:
                choice = console.input("Choose a coin by number or symbol [1]: ").strip()
                if not choice:
                    return symbols[0]
                if choice.isdigit() and 1 <= int(choice) <= len(symbols):
                    return symbols[int(choice) - 1]
                if choice in self.grouped:
                    return choice
                console.print("[red]Invalid selection.[/red]")
        return symbols[0]

    def _prompt_budget(self) -> float:
        console = self._console()
        if console:
            while True:
                value = console.input("Starting budget in USD [10000]: ").strip()
                if not value:
                    return 10_000.0
                try:
                    budget = float(value)
                except ValueError:
                    console.print("[red]Enter a numeric budget.[/red]")
                    continue
                if budget <= 0:
                    console.print("[red]Budget must be greater than zero.[/red]")
                    continue
                return budget
        return 10_000.0

    def _prompt_command(self) -> str:
        console = self._console()
        if console is None:
            return "quit"
        return console.input("[bold cyan]memelearn[/bold cyan] > ").strip()

    def _handle_command(self, command: str) -> bool:
        if not command:
            self._render_screen()
            return True

        parts = command.split()
        action = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else None

        if action in {"help", "?"}:
            self._print_help()
            return True
        if action in {"chart", "c"}:
            self._render_screen("Chart refreshed.")
            return True
        if action in {"status", "s"}:
            self._render_screen("Status refreshed.")
            return True
        if action in {"buy", "b"}:
            self._buy(argument)
            return True
        if action in {"sell", "x"}:
            self._sell(argument)
            return True
        if action in {"next", "n"}:
            if self.current_index >= len(self.selected_bars) - 1:
                self._render_screen("End of data reached. Staying on the last candle.")
                return True
            self.current_index = min(self.current_index + 1, len(self.selected_bars) - 1)
            self._append_equity_point("next")
            self._render_screen("Moved to the next candle.")
            return True
        if action in {"quit", "exit", "q"}:
            return False
        self._render_screen(f"Unknown command: {action}")
        return True

    def _buy(self, argument: str | None) -> bool:
        if self.state is None:
            return False
        bar = self.selected_bars[self.current_index]
        price = self._apply_slippage(bar.close, is_buy=True)
        spend = self._resolve_spend(argument, self.state.cash)
        if spend <= 0:
            self._render_screen("Nothing to buy.")
            return False
        spend = min(spend, self.state.cash / (1.0 + self.backtest_config.fee_rate))
        quantity = safe_div(spend, price, 0.0)
        if quantity <= 0:
            self._render_screen("Buy size is too small.")
            return False
        entry_notional = quantity * price
        fee = entry_notional * self.backtest_config.fee_rate
        total_cost = entry_notional + fee
        if quantity <= 0 or total_cost > self.state.cash + 1e-9:
            self._render_screen("Not enough cash for that buy.")
            return False
        features = self.feature_rows[self.current_index]
        self.state.cash = max(0.0, self.state.cash - total_cost)
        if self.state.position is None:
            self.state.position = OpenPosition(
                symbol=bar.symbol,
                entry_time=bar.timestamp,
                entry_index=self.current_index,
                entry_price=price,
                quantity=quantity,
                stop_loss=price * (1.0 - self.backtest_config.stop_loss_pct),
                take_profit=price * (1.0 + self.backtest_config.take_profit_pct),
                entry_reason="manual_buy",
                entry_features={key: float(value) for key, value in features.items() if isinstance(value, (int, float))},
                entry_tags=["manual"],
                entry_fee=fee,
                max_price=bar.high,
                min_price=bar.low,
            )
            buy_message = f"Bought {quantity:.6f} {bar.symbol} at {format_currency(price)}."
        else:
            self._scale_into_position(
                bar=bar,
                price=price,
                quantity=quantity,
                fee=fee,
                entry_features={key: float(value) for key, value in features.items() if isinstance(value, (int, float))},
            )
            buy_message = f"Added {quantity:.6f} {bar.symbol} at {format_currency(price)}."
        self._append_equity_point("buy")
        self._skip_auto_manage_once = True
        self._render_screen(buy_message)
        return True

    def _scale_into_position(
        self,
        *,
        bar: MarketBar,
        price: float,
        quantity: float,
        fee: float,
        entry_features: dict[str, float],
    ) -> None:
        if self.state is None or self.state.position is None:
            return
        position = self.state.position
        previous_quantity = position.quantity
        previous_notional = previous_quantity * position.entry_price
        added_notional = quantity * price
        total_quantity = previous_quantity + quantity
        total_notional = previous_notional + added_notional

        position.quantity = total_quantity
        position.entry_price = safe_div(total_notional, total_quantity, position.entry_price)
        position.entry_fee += fee
        position.entry_time = min(position.entry_time, bar.timestamp)
        position.entry_index = min(position.entry_index, self.current_index)
        position.stop_loss = position.entry_price * (1.0 - self.backtest_config.stop_loss_pct)
        position.take_profit = position.entry_price * (1.0 + self.backtest_config.take_profit_pct)
        position.max_price = max(position.max_price, bar.high)
        position.min_price = bar.low if position.min_price == 0.0 else min(position.min_price, bar.low)
        position.entry_tags = list(dict.fromkeys(position.entry_tags + ["scale_in"]))
        position.entry_reason = "manual_buy_scale_in"
        position.entry_features = self._merge_entry_features(position.entry_features, entry_features, previous_quantity, quantity)

    def _sell(self, argument: str | None) -> bool:
        if self.state is None or self.state.position is None:
            self._render_screen("No open position to sell.")
            return False
        self._close_position(reason="manual_sell", argument=argument)
        self._append_equity_point("sell")
        return True

    def _close_position(self, reason: str, argument: str | None = None) -> bool:
        if self.state is None or self.state.position is None:
            return False
        bar = self.selected_bars[self.current_index]
        position = self.state.position
        original_quantity = position.quantity
        close_quantity = self._resolve_quantity(argument, position.quantity)
        close_quantity = min(close_quantity, position.quantity)
        if close_quantity <= 0:
            self._render_screen("Sell size is too small.")
            return False
        exit_price = self._apply_slippage(bar.close, is_buy=False)
        gross_proceeds = close_quantity * exit_price
        exit_fee = gross_proceeds * self.backtest_config.fee_rate
        entry_notional = close_quantity * position.entry_price
        allocated_entry_fee = position.entry_fee * safe_div(close_quantity, original_quantity, 0.0)
        gross_pnl = gross_proceeds - entry_notional
        net_pnl = gross_pnl - allocated_entry_fee - exit_fee
        self.state.cash = max(0.0, self.state.cash + gross_proceeds - exit_fee)
        if close_quantity == position.quantity:
            self.state.position = None
            remaining_quantity = 0.0
        else:
            position.quantity -= close_quantity
            position.entry_fee -= allocated_entry_fee
            remaining_quantity = position.quantity
        trade = TradeRecord(
            trade_id=self.state.trade_id,
            symbol=position.symbol,
            entry_time=position.entry_time.isoformat(),
            exit_time=bar.timestamp.isoformat(),
            entry_index=position.entry_index,
            exit_index=self.current_index,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=close_quantity,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=safe_div(net_pnl, entry_notional, 0.0),
            fees=allocated_entry_fee + exit_fee,
            slippage_cost=abs(position.entry_price - exit_price) * close_quantity,
            bars_held=max(1, self.current_index - position.entry_index),
            exit_reason=reason,
            entry_reason=position.entry_reason,
            entry_tags=position.entry_tags,
            entry_features=position.entry_features,
            max_favorable_excursion_pct=safe_div(position.max_price - position.entry_price, position.entry_price, 0.0),
            max_adverse_excursion_pct=safe_div(position.entry_price - position.min_price, position.entry_price, 0.0),
        )
        self.state.trades.append(trade)
        self.state.trade_id += 1
        self._render_screen(
            f"Sold {close_quantity:.6f} {position.symbol} at {format_currency(exit_price)} "
            f"for {format_currency(net_pnl)} net PnL."
        )
        if remaining_quantity > 0:
            self._render_screen(f"Partial position remains: {remaining_quantity:.6f} {position.symbol}.")
        return True

    def _auto_manage_position(self, bar: MarketBar) -> bool:
        if self.state is None or self.state.position is None:
            return False
        position = self.state.position
        position.max_price = max(position.max_price, bar.high)
        position.min_price = bar.low if position.min_price == 0.0 else min(position.min_price, bar.low)
        if bar.low <= position.stop_loss:
            self._close_position(reason="stop_loss")
            return True
        if bar.high >= position.take_profit:
            self._close_position(reason="take_profit")
            return True
        return False

    def _resolve_spend(self, argument: str | None, cash: float) -> float:
        if argument is None or argument.lower() in {"all", "max"}:
            return cash
        if argument.endswith("%"):
            try:
                return cash * clamp(float(argument[:-1]) / 100.0, 0.0, 1.0)
            except ValueError:
                return 0.0
        try:
            return max(0.0, float(argument))
        except ValueError:
            return 0.0

    def _resolve_quantity(self, argument: str | None, available_quantity: float) -> float:
        if argument is None or argument.lower() in {"all", "max"}:
            return available_quantity
        if argument.endswith("%"):
            try:
                return available_quantity * clamp(float(argument[:-1]) / 100.0, 0.0, 1.0)
            except ValueError:
                return 0.0
        try:
            return max(0.0, float(argument))
        except ValueError:
            return 0.0

    def _append_equity_point(self, label: str) -> None:
        if self.state is None:
            return
        bar = self.selected_bars[self.current_index]
        equity = self.state.cash
        position_value = 0.0
        if self.state.position is not None:
            position_value = self.state.position.quantity * bar.close
            equity += position_value
        self.state.equity_curve.append(
            {
                "timestamp": bar.timestamp.isoformat(),
                "equity": equity,
                "cash": self.state.cash,
                "position_value": position_value,
                "label": label,
                "symbol": bar.symbol,
            }
        )

    def _apply_slippage(self, price: float, *, is_buy: bool) -> float:
        rate = self.backtest_config.slippage_rate
        return price * (1.0 + rate if is_buy else 1.0 - rate)

    def _render_screen(self, banner: str | None = None) -> None:
        console = self._console()
        if self.state is None or console is None:
            return

        bar = self.selected_bars[self.current_index]
        window = self.selected_bars[max(0, self.current_index - self.window_size + 1) : self.current_index + 1]
        chart = self._candlestick_chart(window)
        equity = self.state.cash
        position_value = 0.0
        unrealized = 0.0
        if self.state.position is not None:
            position_value = self.state.position.quantity * bar.close
            equity += position_value
            unrealized = (bar.close - self.state.position.entry_price) * self.state.position.quantity

        console.clear()
        console.rule(f"[bold cyan]Memelearn Live Session - {self.symbol}")
        if banner:
            console.print(f"[bold yellow]{banner}[/bold yellow]")
        metrics = Table(show_header=False, box=None)
        metrics.add_column("Field", style="bold")
        metrics.add_column("Value")
        metrics.add_row("Budget", format_currency(self.initial_budget or 0.0))
        metrics.add_row("Cash", format_currency(self.state.cash))
        metrics.add_row("Equity", format_currency(equity))
        metrics.add_row("Open position", "Yes" if self.state.position else "No")
        metrics.add_row("Unrealized PnL", format_currency(unrealized))
        metrics.add_row("Candle", f"{bar.timestamp.isoformat()}")
        metrics.add_row("Close", format_currency(bar.close))
        metrics.add_row("Range", f"{format_currency(bar.low)} - {format_currency(bar.high)}")
        metrics.add_row("Volume", f"{bar.volume:,.0f}")
        metrics.add_row("Candle", f"{self.current_index + 1}/{len(self.selected_bars)}")
        if bar.liquidity is not None:
            metrics.add_row("Liquidity", f"{bar.liquidity:,.0f}")
        pattern_type = bar.raw.get("chart_type") or bar.raw.get("scenario")
        if pattern_type:
            metrics.add_row("Pattern", str(pattern_type))
        console.print(metrics)
        console.print(Panel(chart, title="Candlestick Chart", border_style="cyan"))
        if self.state.position is not None:
            pos = self.state.position
            position_table = Table(title="Open Position")
            position_table.add_column("Field")
            position_table.add_column("Value")
            position_table.add_row("Entry", format_currency(pos.entry_price))
            position_table.add_row("Quantity", f"{pos.quantity:.6f}")
            position_table.add_row("Stop Loss", format_currency(pos.stop_loss))
            position_table.add_row("Take Profit", format_currency(pos.take_profit))
            position_table.add_row("Bars Held", str(self.current_index - pos.entry_index))
            position_table.add_row("Unrealized", format_currency(unrealized))
            console.print(position_table)
        self._print_commands()

    def _print_commands(self) -> None:
        console = self._console()
        if console is None:
            return
        console.print(
            "[bold]Commands:[/bold] "
            "[green]buy[/green] [amount|%|all], "
            "[red]sell[/red] [amount|%|all], "
            "[cyan]next[/cyan], "
            "[cyan]chart[/cyan], "
            "[cyan]status[/cyan], "
            "[cyan]help[/cyan], "
            "[cyan]quit[/cyan]"
        )

    def _print_help(self) -> None:
        console = self._console()
        if console is None:
            return
        console.print(
            "buy 1000    buy with $1000\n"
            "buy 25%     buy using 25% of available cash\n"
            "sell all    close the open position\n"
            "sell 50%    close half the open position\n"
            "next        move to the next candle\n"
            "chart       redraw the chart\n"
            "status      redraw the portfolio status\n"
            "quit        exit and save the session report"
        )

    def _candlestick_chart(self, bars: list[MarketBar], height: int = 10) -> Any:
        if not bars:
            return "No chart data available."
        if Text is None:
            return self._candlestick_chart_plain(bars, height=height)

        max_price = max(bar.high for bar in bars)
        min_price = min(bar.low for bar in bars)
        if abs(max_price - min_price) < 1e-12:
            max_price += 1.0
            min_price -= 1.0

        width = len(bars) * 3
        grid: list[list[tuple[str, str | None]]] = [[(" ", None) for _ in range(width)] for _ in range(height)]

        for index, bar in enumerate(bars):
            color = "green" if bar.close >= bar.open else "red"
            x = index * 3 + 1
            high_row = self._price_to_row(bar.high, min_price, max_price, height)
            low_row = self._price_to_row(bar.low, min_price, max_price, height)
            open_row = self._price_to_row(bar.open, min_price, max_price, height)
            close_row = self._price_to_row(bar.close, min_price, max_price, height)
            top_body = min(open_row, close_row)
            bottom_body = max(open_row, close_row)

            for row in range(high_row, low_row + 1):
                grid[row][x] = ("│", color)
            for row in range(top_body, bottom_body + 1):
                grid[row][x] = ("█", color)
            if open_row == close_row:
                grid[open_row][x] = ("■", color)
            else:
                grid[open_row][x] = ("┤", color)
                grid[close_row][x] = ("├", color)

        chart = Text()
        for row_index, row in enumerate(grid):
            level = max_price - ((max_price - min_price) * row_index / max(height - 1, 1))
            chart.append(f"{level:>8.4f} ")
            for char, color in row:
                if color:
                    chart.append(char, style=color)
                else:
                    chart.append(char)
            chart.append("\n")
        chart.append("         Legend: ")
        chart.append("green", style="green")
        chart.append(" = bullish, ")
        chart.append("red", style="red")
        chart.append(" = bearish\n")
        return chart

    def _candlestick_chart_plain(self, bars: list[MarketBar], height: int = 10) -> str:
        max_price = max(bar.high for bar in bars)
        min_price = min(bar.low for bar in bars)
        if abs(max_price - min_price) < 1e-12:
            max_price += 1.0
            min_price -= 1.0
        width = len(bars) * 3
        grid = [[" " for _ in range(width)] for _ in range(height)]
        for index, bar in enumerate(bars):
            x = index * 3 + 1
            high_row = self._price_to_row(bar.high, min_price, max_price, height)
            low_row = self._price_to_row(bar.low, min_price, max_price, height)
            open_row = self._price_to_row(bar.open, min_price, max_price, height)
            close_row = self._price_to_row(bar.close, min_price, max_price, height)
            top_body = min(open_row, close_row)
            bottom_body = max(open_row, close_row)
            for row in range(high_row, low_row + 1):
                grid[row][x] = "|"
            for row in range(top_body, bottom_body + 1):
                grid[row][x] = "#"
        lines = []
        for row_index, row in enumerate(grid):
            level = max_price - ((max_price - min_price) * row_index / max(height - 1, 1))
            lines.append(f"{level:>8.4f} {''.join(row)}")
        return "\n".join(lines)

    def _price_to_row(self, price: float, min_price: float, max_price: float, height: int) -> int:
        normalized = clamp((max_price - price) / max(max_price - min_price, 1e-12), 0.0, 1.0)
        return int(round(normalized * (height - 1)))

    def _merge_entry_features(
        self,
        current: dict[str, float],
        new: dict[str, float],
        previous_quantity: float,
        added_quantity: float,
    ) -> dict[str, float]:
        if not current:
            return dict(new)
        total = previous_quantity + added_quantity
        if total <= 0:
            return dict(new)
        merged = dict(current)
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
                continue
            merged[key] = safe_div(merged[key] * previous_quantity + value * added_quantity, total, value)
        return merged

    def _note(self, message: str) -> None:
        self._render_screen(message)

    def _finalize_session(self) -> None:
        if self.state is None:
            return
        if self.state.position is not None:
            self._close_position(reason="end_of_session")
        if self.current_index < len(self.selected_bars):
            self._append_equity_point("session_end")

        result = BacktestResult(
            strategy_name=f"InteractiveSession:{self.symbol}",
            config=self.backtest_config.to_dict(),
            data_summary=summarize_market_data(self.selected_bars).to_dict(),
            trades=self.state.trades,
            equity_curve=self.state.equity_curve,
            final_equity=self.state.cash,
            initial_capital=float(self.initial_budget or 0.0),
            notes=["Interactive session started from the launcher."],
        )
        metrics = compute_metrics(result)
        score = score_strategy(metrics, result.trades, len(result.equity_curve))
        failure_report = analyze_failures(result.trades) if result.trades else None
        model = train_pattern_model(result.trades) if len(result.trades) >= 2 else None
        render_backtest_console(result, metrics.metrics, score, failure_report, model)
        save_backtest_artifacts(self.output_dir, result, metrics.metrics, score, failure_report, model)


def run_interactive_session(
    *,
    data_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs/interactive",
    symbol: str | None = None,
    budget: float | None = None,
    window_size: int = 20,
) -> None:
    config_payload = load_app_config(config_path) if config_path else load_app_config(None)
    backtest_config = backtest_config_from_dict(config_payload["backtest"])
    strategy_config = strategy_config_from_dict(config_payload["strategy"])
    bars = load_market_data(data_path)
    # Build the strategy now so the interactive session can evolve into strategy-assisted play later.
    _ = MemeCoinStrategy(strategy_config)
    session = InteractiveTraderSession(
        bars,
        output_dir=output_dir,
        config=backtest_config,
        window_size=window_size,
        symbol=symbol,
        budget=budget,
    )
    session.run()
