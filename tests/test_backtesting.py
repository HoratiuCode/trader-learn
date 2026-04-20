from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_backtesting.config import backtest_config_from_dict, strategy_config_from_dict
from trader_backtesting.data_loading import load_market_data, summarize_market_data
from trader_backtesting.metrics import compute_metrics
from trader_backtesting.ml_analysis import train_pattern_model
from trader_backtesting.pattern_analysis import analyze_failures
from trader_backtesting.scoring import score_strategy
from trader_backtesting.simulator import Backtester
from trader_backtesting.strategies import MemeCoinStrategy


class BacktestingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_csv = ROOT / "data" / "sample_market_data.csv"
        if not sample_csv.exists():
            raise unittest.SkipTest("Sample dataset has not been generated yet.")

    def test_load_and_summarize(self) -> None:
        bars = load_market_data(ROOT / "data" / "sample_market_data.csv")
        summary = summarize_market_data(bars)
        self.assertGreater(summary.count, 0)
        self.assertEqual(summary.symbols, ["MEME"])

    def test_backtest_pipeline(self) -> None:
        bars = load_market_data(ROOT / "data" / "sample_market_data.csv")
        config = backtest_config_from_dict(
            {
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
            }
        )
        strategy = MemeCoinStrategy(strategy_config_from_dict({}))
        result = Backtester(config, strategy).run(bars)
        metrics = compute_metrics(result)
        score = score_strategy(metrics, result.trades, len(result.equity_curve))
        self.assertGreater(len(result.trades), 0)
        self.assertIn("total_return", metrics.metrics)
        self.assertGreaterEqual(score.total_score, 0.0)

    def test_analysis_and_model(self) -> None:
        bars = load_market_data(ROOT / "data" / "sample_market_data.csv")
        config = backtest_config_from_dict({})
        strategy = MemeCoinStrategy(strategy_config_from_dict({}))
        result = Backtester(config, strategy).run(bars)
        failure_report = analyze_failures(result.trades)
        model = train_pattern_model(result.trades)
        self.assertTrue(failure_report.recommendations)
        self.assertTrue(model.feature_rankings)


if __name__ == "__main__":
    unittest.main()
