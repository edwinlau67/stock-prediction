# CLAUDE.md

We're building the app described in @DESIGN.md. Read that file for general architectural tasks or to double-check the exact database structure, tech stack or application architecture.

Keep your replies extremely concise and focus on conveying the key information. No unnecessary fluff, no long code snippets.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock Predictor is a single-file Python CLI that uses Claude's tool-use to generate technical and fundamental stock analysis. Claude calls a `stock_prediction` tool, which fetches live data from Yahoo Finance, computes 20+ indicators, scores them, and returns a prediction dict. Claude then writes a Markdown report with embedded charts.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Run (defaults: AAPL 1w, TSLA 1m, INTC 1m)
python stock_predictor.py

# Common usage patterns
python stock_predictor.py --tickers NVDA --timeframe 3m
python stock_predictor.py --tickers AAPL MSFT --timeframe 1m --indicators trend momentum
python stock_predictor.py --tickers TSLA --model claude-opus-4-7
python stock_predictor.py --tickers AAPL --config config/scoring_config.json
python stock_predictor.py --tickers NVDA --log-level DEBUG

# Run tests
python -m pytest tests/
```

**CLI arguments:**
- `--tickers` — one or more symbols
- `--timeframe` — `1d | 1w | 1m | 3m | 6m | ytd | 1y | 2y | 5y`
- `--model` — Claude model ID (default: `claude-sonnet-4-6`)
- `--indicators` — space-separated subset of: `trend momentum volatility volume support fundamental`
- `--config` — path to a JSON file with `ScoringConfig` threshold overrides; default file at `config/scoring_config.json`
- `--log-level` — `DEBUG | INFO | WARNING` (default: `INFO`)

To verify a change works, run against a single ticker and check `results/*/predictions.md` and `results/*/charts/`. The test suite is in `tests/test_stock_predictor.py`.

## Architecture

All code lives in `stock_predictor.py`. The flow:

1. **CLI → `predict_stock()`** — top-level orchestrator per ticker. Builds the Anthropic API call with a system prompt (prompt-cached) and the `stock_prediction` tool schema.

2. **Claude tool-use loop** — Claude calls `stock_prediction` with ticker + timeframe. The handler calls `run_prediction()`, which:
   - Fetches current price via `get_current_price()` (yfinance `fast_info`)
   - Fetches 1-year OHLCV via `get_technical_indicators()` (with `_fetch_with_retry` for resilience)
   - Fetches company metrics via `get_fundamental_indicators()`
   - Delegates scoring to six `_score_*()` helpers (one per category), each accepting a `ScoringConfig`
   - Accumulates `bullish_score` / `bearish_score`, derives direction, confidence, price target, and risk level
   - Returns a prediction dict to Claude

3. **Claude response** — Claude formats the prediction as Markdown using the strict template defined in the system prompt (fixed section headers, emoji, tables).

4. **Post-processing** — `generate_chart()` delegates each panel to a `_draw_*()` helper and builds the dynamic `GridSpec` (rows added only for selected indicator categories, dark theme). The chart path is embedded in the Markdown. Results are written to `results/YYYYMMDD_HHMMSS/`.

### Prediction dict schema

```python
{
    "ticker": str,
    "timeframe": str,
    "direction": "bullish" | "bearish" | "neutral",
    "confidence": float,        # 0.52–0.95
    "current_price": float,
    "price_target": float,
    "target_date": str,         # ISO date
    "key_factors": list[str],   # up to 6 signals
    "risk_level": "low" | "medium" | "high",
    "technical": dict,          # full indicator arrays
    "fundamental": dict,        # 15 metrics
    "indicators": list[str],    # sorted active categories
}
```

### ScoringConfig

All scoring thresholds are centralised in the `ScoringConfig` dataclass (fundamental cutoffs, RSI/Stochastic levels, ATR ratios, confidence formula constants, price-target ranges). Load overrides at runtime with `--config thresholds.json`. Defaults match the original hard-coded values.

### Scoring model

Each indicator contributes to `bullish_score` or `bearish_score` via `_score_trend()`, `_score_momentum()`, `_score_volatility()`, `_score_volume()`, `_score_support()`, and `_score_fundamental()`. Direction is whichever score is higher. Confidence = `min(cfg.conf_cap, cfg.conf_base + gap × cfg.conf_gap_factor + random(0, cfg.conf_noise_max))`. Risk level is derived from ATR ratio vs. 20-day mean (>1.3× = high, <0.8× = low).

### Prompt caching

The system prompt is sent with `cache_control: {type: "ephemeral"}` for 5-minute prefix caching. This reduces token costs on multi-ticker runs.

## Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic==0.96.0` | Claude API, tool use, prompt caching |
| `yfinance==1.3.0` | Live prices, OHLCV history, fundamentals |
| `matplotlib==3.10.8` | Multi-panel chart rendering |
| `numpy==2.4.0` | Indicator calculations |

Python 3.8+ required; tested on 3.13.7.
