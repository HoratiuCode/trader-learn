# trader-backtesting

Terminal-first backtesting toolkit for meme coin strategies.

## What it does

- Loads historical token OHLCV data from CSV or JSON
- Simulates long-only buying and selling
- Tracks cash, equity, PnL, fees, slippage, drawdown, and win rate
- Scores strategy quality across profitability, consistency, risk control, drawdown, timing, and discipline
- Clusters losing trades into failure patterns
- Trains a lightweight ML-style feature importance model from trade outcomes
- Writes reports and JSON summaries to disk

## Project layout

- `src/trader_backtesting/` core package
- `data/` sample datasets
- `config/` sample configuration
- `scripts/` helper scripts
- `tests/` unit tests
- `outputs/` generated reports

## Setup

The project is designed for Python 3.11+.

```bash
cd /Users/horatiubudai/ceo/Hacker/trenchlearn/trader-backtesting
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If `rich` is unavailable, the CLI still runs with plain text output, but the recommended setup installs the dependencies above.

## Generate sample assets

The repo includes a deterministic generator for the sample dataset and config:

```bash
python scripts/generate_sample_data.py
```

That writes:

- `data/sample_market_data.csv`
- `data/sample_market_data.json`
- `config/sample_config.json`

## Example commands

Load and validate data:

```bash
trader-backtesting load-data --input data/sample_market_data.csv --output-dir outputs/load-data
```

Run the sample backtest:

```bash
trader-backtesting backtest --data data/sample_market_data.csv --config config/sample_config.json --output-dir outputs/backtest
```

Start the interactive memelearn session:

```bash
trader-backtesting start --data data/sample_market_data.csv --config config/sample_config.json --output-dir outputs/interactive
```

Render a saved report:

```bash
trader-backtesting report --summary outputs/backtest/backtest_summary.json
```

Analyze losing trades:

```bash
trader-backtesting analyze-failures --summary outputs/backtest/backtest_summary.json --output-dir outputs/failures
```

Train the ML helper model:

```bash
trader-backtesting train-model --summary outputs/backtest/backtest_summary.json --output-dir outputs/ml
```

Show pattern suggestions:

```bash
trader-backtesting suggest-patterns --model outputs/ml/ml_model.json
```

## CLI commands

- `load-data`
- `backtest`
- `start`
- `report`
- `analyze-failures`
- `train-model`
- `suggest-patterns`

## Default launcher

Double-click `~/Documents/memelearn.command` to:

- open the project folder
- start the interactive session
- choose a budget
- select a memecoin
- trade with `buy`, `sell`, `next`, and `chart`

## Example dataset format

The loader accepts rows with:

- `timestamp`
- `symbol` or `token`
- `open`
- `high`
- `low`
- `close`
- `volume`
- optional `liquidity`
- optional `market_cap`

CSV example:

```csv
timestamp,symbol,open,high,low,close,volume,liquidity,market_cap
2026-04-20T00:00:00+00:00,MEME,0.99,1.02,0.98,1.00,1000,50000,92500
```

The bundled sample dataset now includes multiple memecoins with different behavior profiles:

- `PUPILO` - trend breakout
- `DECODE` - fake breakout
- `COFFE67` - vertical pump and dump
- `NULLO` - unstable reversal
- `VIBE7` - trend pullback continuation
- `GLITCH` - chop
- `BOUNCE9` - breakout continuation
- `ZAZA` - exhaustion recovery
- `MEME` - breakout continuation

## Architecture

The code is intentionally modular:

- `data_loading` parses and normalizes CSV/JSON market data
- `features` computes rolling technical and meme-coin-specific features
- `strategies` contains the strategy interface and sample strategy
- `simulator` executes the backtest and records trades
- `metrics` computes performance statistics
- `scoring` converts metrics into a trader/strategy score
- `pattern_analysis` groups losing trades and describes failure modes
- `ml_analysis` builds a lightweight baseline model and feature importance report
- `reporting` renders terminal and file-based reports
- `cli` ties everything together into a terminal app

## Example failure analysis

A typical output from the sample dataset looks like:

- `strategy fails most during low-liquidity fake breakouts`
- `strategy loses often after 3-candle vertical pumps`
- `entries are too early during unstable reversals`

## Example ML suggestions

A typical output from the sample dataset looks like:

- `Collect more examples of fast pump-then-dump patterns.`
- `Add features for volume exhaustion.`
- `Separate trend days from chop days.`

## Tests

Run the core tests with:

```bash
python3.11 -m unittest discover -s tests -v
```

## Future improvements

- Multi-asset portfolio backtesting
- Walk-forward evaluation and parameter sweeps
- More robust ML classifiers
- Slippage models based on liquidity and volatility
- Better regime detection for chop versus trend days
- Real-time trade journaling and strategy comparison
