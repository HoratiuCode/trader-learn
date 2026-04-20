from __future__ import annotations

import argparse
from pathlib import Path

from .config import backtest_config_from_dict, load_app_config, strategy_config_from_dict
from .data_loading import load_market_data, save_normalized_market_data, summarize_market_data
from .interactive import run_interactive_session
from .metrics import compute_metrics
from .ml_analysis import train_pattern_model
from .pattern_analysis import analyze_failures
from .reporting import (
    build_failure_markdown,
    build_ml_markdown,
    render_backtest_console,
    save_backtest_artifacts,
)
from .scoring import score_strategy
from .simulator import Backtester
from .strategies import MemeCoinStrategy
from .utils import ensure_dir, read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trader-backtesting", description="Meme coin strategy backtesting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_cmd = subparsers.add_parser("load-data", help="Load and validate CSV or JSON market data")
    load_cmd.add_argument("--input", required=True, help="Path to CSV or JSON data file")
    load_cmd.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts")

    backtest_cmd = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_cmd.add_argument("--data", required=True, help="Path to CSV or JSON market data")
    backtest_cmd.add_argument("--config", help="Optional JSON config file")
    backtest_cmd.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts")

    start_cmd = subparsers.add_parser("start", help="Start the interactive memelearn trading session")
    start_cmd.add_argument("--data", default="data/sample_market_data.csv", help="Path to CSV or JSON market data")
    start_cmd.add_argument("--config", default="config/sample_config.json", help="Optional JSON config file")
    start_cmd.add_argument("--output-dir", default="outputs/interactive", help="Directory for generated artifacts")
    start_cmd.add_argument("--symbol", help="Preselect a symbol instead of prompting")
    start_cmd.add_argument("--budget", type=float, help="Preselect a starting budget instead of prompting")
    start_cmd.add_argument("--window-size", type=int, default=20, help="Candles to show in the live chart")

    report_cmd = subparsers.add_parser("report", help="Render a report from a saved summary JSON")
    report_cmd.add_argument("--summary", required=True, help="Path to backtest_summary.json")

    failure_cmd = subparsers.add_parser("analyze-failures", help="Analyze losing trades from a saved summary JSON")
    failure_cmd.add_argument("--summary", required=True, help="Path to backtest_summary.json")
    failure_cmd.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts")

    model_cmd = subparsers.add_parser("train-model", help="Train the lightweight ML pattern model")
    model_cmd.add_argument("--summary", required=True, help="Path to backtest_summary.json")
    model_cmd.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts")

    suggest_cmd = subparsers.add_parser("suggest-patterns", help="Show ML-driven pattern suggestions")
    suggest_cmd.add_argument("--model", required=True, help="Path to ml_model.json")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command.replace("-", "_")
    handler = globals().get(f"cmd_{command}")
    if handler is None:
        raise SystemExit(f"Unknown command: {args.command}")
    handler(args)


def cmd_load_data(args: argparse.Namespace) -> None:
    bars = load_market_data(args.input)
    summary = summarize_market_data(bars)
    output_dir = ensure_dir(args.output_dir)
    save_normalized_market_data(output_dir / "normalized_market_data.json", bars)
    write_json(output_dir / "data_summary.json", summary.to_dict())
    print(f"Loaded {summary.count} bars across {len(summary.symbols)} symbol(s).")
    print(f"Normalized data written to {output_dir / 'normalized_market_data.json'}")


def cmd_backtest(args: argparse.Namespace) -> None:
    config_payload = load_app_config(args.config) if args.config else load_app_config(None)
    backtest_config = backtest_config_from_dict(config_payload["backtest"])
    strategy_config = strategy_config_from_dict(config_payload["strategy"])
    bars = load_market_data(args.data)
    result, metrics_bundle, score, failure_report, model = _run_analysis_pipeline(bars, backtest_config, strategy_config)
    render_backtest_console(result, metrics_bundle.metrics, score, failure_report, model)
    paths = save_backtest_artifacts(args.output_dir, result, metrics_bundle.metrics, score, failure_report, model)
    print(f"Saved summary to {paths['summary_json']}")
    print(f"Saved trade log to {paths['trade_log_json']}")
    print(f"Saved report to {paths['report_md']}")


def cmd_start(args: argparse.Namespace) -> None:
    run_interactive_session(
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output_dir,
        symbol=args.symbol,
        budget=args.budget,
        window_size=args.window_size,
    )


def cmd_report(args: argparse.Namespace) -> None:
    summary = read_json(args.summary)
    result = summary["result"]
    metrics = summary["metrics"]
    score = summary["score"]
    from .models import BacktestResult, ScoreBreakdown, TradeRecord

    trades = [TradeRecord(**trade) for trade in result["trades"]]
    backtest_result = BacktestResult(
        strategy_name=result["strategy_name"],
        config=result["config"],
        data_summary=result["data_summary"],
        trades=trades,
        equity_curve=result["equity_curve"],
        final_equity=result["final_equity"],
        initial_capital=result["initial_capital"],
        notes=result.get("notes", []),
    )
    score_obj = ScoreBreakdown(**score)
    failure_report = None
    if summary.get("failure_report"):
        from .models import FailureAnalysisReport, FailureCluster

        failure = summary["failure_report"]
        failure_report = FailureAnalysisReport(
            summary=failure["summary"],
            clusters=[FailureCluster(**cluster) for cluster in failure["clusters"]],
            recommendations=failure["recommendations"],
        )
    model = None
    if summary.get("model"):
        from .models import ModelArtifact

        model_payload = summary["model"]
        model = ModelArtifact(
            model_name=model_payload["model_name"],
            bias=model_payload["bias"],
            feature_stats=model_payload["feature_stats"],
            feature_weights=model_payload["feature_weights"],
            label_stats=model_payload["label_stats"],
            pattern_summary=model_payload["pattern_summary"],
            feature_rankings=model_payload["feature_rankings"],
            recommendations=model_payload["recommendations"],
        )
    render_backtest_console(backtest_result, metrics, score_obj, failure_report, model)


def cmd_analyze_failures(args: argparse.Namespace) -> None:
    summary = read_json(args.summary)
    result = summary["result"]
    from .models import TradeRecord

    trades = [TradeRecord(**trade) for trade in result["trades"]]
    report = analyze_failures(trades)
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "failure_report.json", report.to_dict())
    (output_dir / "failure_report.md").write_text(build_failure_markdown(report), encoding="utf-8")
    print(build_failure_markdown(report))
    print(f"Saved failure report to {output_dir / 'failure_report.json'}")


def cmd_train_model(args: argparse.Namespace) -> None:
    summary = read_json(args.summary)
    result = summary["result"]
    from .models import TradeRecord

    trades = [TradeRecord(**trade) for trade in result["trades"]]
    model = train_pattern_model(trades)
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "ml_model.json", model.to_dict())
    write_json(output_dir / "ml_report.json", {"model": model.to_dict()})
    (output_dir / "ml_report.md").write_text(build_ml_markdown(model), encoding="utf-8")
    print(build_ml_markdown(model))
    print(f"Saved model to {output_dir / 'ml_model.json'}")


def cmd_suggest_patterns(args: argparse.Namespace) -> None:
    payload = read_json(args.model)
    from .models import ModelArtifact

    model = ModelArtifact(
        model_name=payload["model_name"],
        bias=payload["bias"],
        feature_stats=payload["feature_stats"],
        feature_weights=payload["feature_weights"],
        label_stats=payload["label_stats"],
        pattern_summary=payload["pattern_summary"],
        feature_rankings=payload["feature_rankings"],
        recommendations=payload["recommendations"],
    )
    print("Pattern suggestions:")
    for recommendation in model.recommendations:
        print(f" - {recommendation}")
    print("Top features:")
    for item in model.feature_rankings[:10]:
        print(f" - {item['feature']}: weight={item['weight']:.4f}")


def _run_analysis_pipeline(bars, backtest_config, strategy_config):
    strategy = MemeCoinStrategy(strategy_config)
    runner = Backtester(backtest_config, strategy)
    result = runner.run(bars)
    metrics_bundle = compute_metrics(result)
    score = score_strategy(metrics_bundle, result.trades, len(result.equity_curve))
    failure_report = analyze_failures(result.trades)
    model = train_pattern_model(result.trades) if len(result.trades) >= 2 else None
    return result, metrics_bundle, score, failure_report, model
