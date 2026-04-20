from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - graceful fallback
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

from .models import BacktestResult, FailureAnalysisReport, ModelArtifact, ScoreBreakdown, TradeRecord
from .utils import ensure_dir, format_currency, format_pct, write_json, write_text


def _console() -> Any:
    if Console is None:
        return None
    return Console()


def render_backtest_console(
    result: BacktestResult,
    metrics: dict[str, float],
    score: ScoreBreakdown,
    failure_report: FailureAnalysisReport | None = None,
    model: ModelArtifact | None = None,
) -> None:
    console = _console()
    if console is None:
        _print_plain(result, metrics, score, failure_report, model)
        return

    console.rule(f"[bold cyan]Backtest Summary: {result.strategy_name}")
    summary = Table(show_header=False, box=None)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Initial capital", format_currency(result.initial_capital))
    summary.add_row("Final equity", format_currency(result.final_equity))
    summary.add_row("Total return", format_pct(metrics["total_return"]))
    summary.add_row("Win rate", format_pct(metrics["win_rate"]))
    summary.add_row("Max drawdown", format_pct(metrics["max_drawdown"]))
    summary.add_row("Sharpe-like", f"{metrics['sharpe_like']:.2f}")
    summary.add_row("Profit factor", f"{metrics['profit_factor']:.2f}")
    summary.add_row("Score", f"{score.total_score:.2f} / 100")
    console.print(summary)
    console.print()

    trade_table = Table(title="Per-Trade Log")
    columns = [
        "ID",
        "Entry",
        "Exit",
        "Net PnL",
        "Return",
        "Bars",
        "Reason",
    ]
    for column in columns:
        trade_table.add_column(column)
    for trade in result.trades[:20]:
        trade_table.add_row(
            str(trade.trade_id),
            trade.entry_time,
            trade.exit_time,
            format_currency(trade.net_pnl),
            format_pct(trade.return_pct),
            str(trade.bars_held),
            trade.exit_reason,
        )
    console.print(trade_table)

    if failure_report is not None:
        console.print()
        _render_failure_console(console, failure_report)
    if model is not None:
        console.print()
        _render_model_console(console, model)


def _render_failure_console(console: Any, report: FailureAnalysisReport) -> None:
    console.rule("[bold red]Failure Analysis")
    console.print(report.summary)
    table = Table(title="Failure Clusters")
    table.add_column("Cluster")
    table.add_column("Label")
    table.add_column("Count", justify="right")
    table.add_column("Loss", justify="right")
    for cluster in report.clusters[:10]:
        table.add_row(
            cluster.key,
            cluster.label,
            str(cluster.count),
            format_currency(cluster.total_loss),
        )
    console.print(table)
    console.print("Recommendations:")
    for recommendation in report.recommendations:
        console.print(f" - {recommendation}")


def _render_model_console(console: Any, model: ModelArtifact) -> None:
    console.rule("[bold green]ML Analysis")
    console.print(f"Model: {model.model_name}")
    table = Table(title="Feature Importance")
    table.add_column("Feature")
    table.add_column("Weight", justify="right")
    table.add_column("Importance", justify="right")
    for item in model.feature_rankings[:10]:
        table.add_row(item["feature"], f"{item['weight']:.4f}", f"{item['importance']:.4f}")
    console.print(table)
    console.print("Pattern summary:")
    for pattern, stats in sorted(model.pattern_summary.items(), key=lambda item: item[1]["win_rate"]):
        console.print(f" - {pattern}: win_rate={stats['win_rate']:.2f}, count={stats['count']}")
    console.print("Recommendations:")
    for recommendation in model.recommendations:
        console.print(f" - {recommendation}")


def _print_plain(
    result: BacktestResult,
    metrics: dict[str, float],
    score: ScoreBreakdown,
    failure_report: FailureAnalysisReport | None = None,
    model: ModelArtifact | None = None,
) -> None:
    print(f"Backtest Summary: {result.strategy_name}")
    print(f"Initial capital: {format_currency(result.initial_capital)}")
    print(f"Final equity: {format_currency(result.final_equity)}")
    print(f"Total return: {format_pct(metrics['total_return'])}")
    print(f"Win rate: {format_pct(metrics['win_rate'])}")
    print(f"Max drawdown: {format_pct(metrics['max_drawdown'])}")
    print(f"Sharpe-like: {metrics['sharpe_like']:.2f}")
    print(f"Score: {score.total_score:.2f} / 100")
    print("Trades:")
    for trade in result.trades[:20]:
        print(
            f"  #{trade.trade_id} {trade.entry_time} -> {trade.exit_time} "
            f"net={format_currency(trade.net_pnl)} return={format_pct(trade.return_pct)} reason={trade.exit_reason}"
        )
    if failure_report is not None:
        print("Failure analysis:")
        print(failure_report.summary)
        for cluster in failure_report.clusters[:10]:
            print(f"  - {cluster.label} [{cluster.count}] {format_currency(cluster.total_loss)}")
        for recommendation in failure_report.recommendations:
            print(f"  * {recommendation}")
    if model is not None:
        print("ML analysis:")
        for item in model.feature_rankings[:10]:
            print(f"  - {item['feature']}: weight={item['weight']:.4f}")


def save_backtest_artifacts(
    output_dir: str | Path,
    result: BacktestResult,
    metrics: dict[str, float],
    score: ScoreBreakdown,
    failure_report: FailureAnalysisReport | None = None,
    model: ModelArtifact | None = None,
) -> dict[str, str]:
    target = ensure_dir(output_dir)
    paths = {
        "summary_json": str(target / "backtest_summary.json"),
        "trade_log_json": str(target / "trade_log.json"),
        "trade_log_csv": str(target / "trade_log.csv"),
        "report_md": str(target / "backtest_report.md"),
    }
    if failure_report is not None:
        paths["failure_report_json"] = str(target / "failure_report.json")
        paths["failure_report_md"] = str(target / "failure_report.md")
    if model is not None:
        paths["model_json"] = str(target / "ml_model.json")
        paths["ml_report_json"] = str(target / "ml_report.json")
        paths["ml_report_md"] = str(target / "ml_report.md")

    summary = {
        "result": result.to_dict(),
        "metrics": metrics,
        "score": score.to_dict(),
        "failure_report": failure_report.to_dict() if failure_report else None,
        "model": model.to_dict() if model else None,
    }
    write_json(paths["summary_json"], summary)
    write_json(paths["trade_log_json"], [trade.to_dict() for trade in result.trades])
    _write_trade_csv(paths["trade_log_csv"], result.trades)
    write_text(paths["report_md"], build_backtest_markdown(result, metrics, score, failure_report, model))
    if failure_report is not None:
        write_json(paths["failure_report_json"], failure_report.to_dict())
        write_text(paths["failure_report_md"], build_failure_markdown(failure_report))
    if model is not None:
        write_json(paths["model_json"], model.to_dict())
        write_json(paths["ml_report_json"], {"model": model.to_dict()})
        write_text(paths["ml_report_md"], build_ml_markdown(model))
    return paths


def _write_trade_csv(path: str | Path, trades: list[TradeRecord]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trades[0].to_dict().keys()) if trades else [])
        if trades:
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_dict())


def build_backtest_markdown(
    result: BacktestResult,
    metrics: dict[str, float],
    score: ScoreBreakdown,
    failure_report: FailureAnalysisReport | None = None,
    model: ModelArtifact | None = None,
) -> str:
    lines = [
        f"# Backtest Report - {result.strategy_name}",
        "",
        "## Summary",
        f"- Initial capital: {format_currency(result.initial_capital)}",
        f"- Final equity: {format_currency(result.final_equity)}",
        f"- Total return: {format_pct(metrics['total_return'])}",
        f"- Win rate: {format_pct(metrics['win_rate'])}",
        f"- Max drawdown: {format_pct(metrics['max_drawdown'])}",
        f"- Sharpe-like: {metrics['sharpe_like']:.2f}",
        f"- Score: {score.total_score:.2f} / 100",
        "",
        "## Score Breakdown",
    ]
    for name, value in score.components.items():
        lines.append(f"- {name}: {value:.2f}")
    lines.extend(["", "## Trades"])
    for trade in result.trades:
        lines.append(
            f"- #{trade.trade_id} {trade.entry_time} -> {trade.exit_time} "
            f"net={format_currency(trade.net_pnl)} return={format_pct(trade.return_pct)} reason={trade.exit_reason}"
        )
    if failure_report is not None:
        lines.extend(["", "## Failure Analysis", failure_report.summary])
        for cluster in failure_report.clusters:
            lines.append(f"- {cluster.label} ({cluster.count})")
        lines.append("")
        lines.append("### Recommendations")
        for recommendation in failure_report.recommendations:
            lines.append(f"- {recommendation}")
    if model is not None:
        lines.extend(["", "## ML Analysis"])
        lines.append(f"- Model: {model.model_name}")
        lines.append("### Top Features")
        for item in model.feature_rankings[:10]:
            lines.append(f"- {item['feature']}: weight={item['weight']:.4f}")
        lines.append("### Recommendations")
        for recommendation in model.recommendations:
            lines.append(f"- {recommendation}")
    return "\n".join(lines) + "\n"


def build_failure_markdown(report: FailureAnalysisReport) -> str:
    lines = ["# Failure Analysis", "", report.summary, "", "## Clusters"]
    for cluster in report.clusters:
        lines.append(f"- {cluster.label}: {cluster.count} trades, loss={format_currency(cluster.total_loss)}")
    lines.extend(["", "## Recommendations"])
    for recommendation in report.recommendations:
        lines.append(f"- {recommendation}")
    return "\n".join(lines) + "\n"


def build_ml_markdown(model: ModelArtifact) -> str:
    lines = ["# ML Analysis", "", f"- Model: {model.model_name}", "", "## Feature Importance"]
    for item in model.feature_rankings:
        lines.append(f"- {item['feature']}: weight={item['weight']:.4f}")
    lines.extend(["", "## Pattern Summary"])
    for pattern, stats in sorted(model.pattern_summary.items(), key=lambda item: item[1]["win_rate"]):
        lines.append(f"- {pattern}: win_rate={stats['win_rate']:.2f}, count={stats['count']}")
    lines.extend(["", "## Recommendations"])
    for recommendation in model.recommendations:
        lines.append(f"- {recommendation}")
    return "\n".join(lines) + "\n"
