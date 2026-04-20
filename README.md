# Stock Predictor

A Python application that uses the [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) and Claude's tool use feature to predict stock performance. Claude calls the **Stock Prediction** tool to compute real technical indicators and generate predictions, then delivers a structured analysis with price targets, confidence scores, and trend signals — saved as a Markdown report with analysis charts.

## How It Works

```
CLI args → Claude → Stock Prediction tool call → Live price + technical indicators (yfinance) → Claude analysis → Markdown report + charts
```

1. You specify one or more tickers and a timeframe via CLI arguments
2. Claude invokes the **Stock Prediction** tool with the ticker and timeframe
3. Stock Prediction fetches the **real current price** and computes trend-following indicators (SMA, EMA, MACD) from Yahoo Finance
4. Direction and confidence are derived from real technical signals (Golden/Death Cross, MACD crossovers, price vs MAs)
5. Claude analyzes the data and writes a formatted Markdown report
6. A 3-panel analysis chart (PNG) is generated per ticker
7. All output is saved into a timestamped folder under `results/`

## Requirements

- Python 3.8+
- An [Anthropic API key](https://console.anthropic.com) with available credits

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | 0.96.0 | Claude API SDK and tool use |
| `yfinance` | 1.3.0 | Live stock price data from Yahoo Finance |
| `matplotlib` | 3.10.8 | Analysis chart generation |
| `numpy` | 2.4.4 | Numerical operations for chart rendering |

## Setup

```bash
# Clone or download the project
cd stock-prediction

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Run with defaults

```bash
python stock_predictor.py
```

Runs predictions for **AAPL** (1w), **TSLA** (1m), and **INTC** (1m).

### Specify tickers

```bash
python stock_predictor.py --tickers NVDA
python stock_predictor.py --tickers AAPL TSLA NVDA
```

### Specify tickers and timeframe

```bash
python stock_predictor.py --tickers MSFT --timeframe 3m
python stock_predictor.py --tickers GOOG AMZN --timeframe 1d
python stock_predictor.py --tickers NVDA --timeframe 6m
python stock_predictor.py --tickers AAPL --model claude-opus-4-7
```

### All options

```
usage: stock_predictor.py [-h] [--tickers TICKER [TICKER ...]]
                          [--timeframe {1d,1w,1m,3m,6m}]
                          [--model MODEL]

options:
  --tickers   One or more stock ticker symbols (default: AAPL TSLA INTC)
  --timeframe Prediction timeframe for all tickers: 1d, 1w, 1m, 3m, 6m (default: 1w)
  --model     Claude model ID to use (default: claude-sonnet-4-6)
```

### Supported timeframes

| Value | Meaning  |
|-------|----------|
| `1d`  | 1 day    |
| `1w`  | 1 week   |
| `1m`  | 1 month  |
| `3m`  | 3 months |
| `6m`  | 6 months |

### Supported models

| Model ID | Notes |
|----------|-------|
| `claude-sonnet-4-6` | Default — fast and cost-effective |
| `claude-opus-4-7` | Most capable, higher cost |
| `claude-haiku-4-5-20251001` | Fastest and cheapest |

## Output

Each run creates a timestamped folder under `results/`:

```
results/
└── 20260419_041029/
    ├── predictions.md       ← Markdown report with embedded chart images
    └── charts/
        ├── AAPL_1w.png
        ├── TSLA_1m.png
        └── INTC_1m.png
```

### Markdown report

The `predictions.md` file contains a full analysis per ticker including a summary table, key factors, and AI narrative — with each chart embedded inline. The report header records the tickers, timeframe, and model used.

### Analysis charts

Each PNG chart contains 6 panels across 4 rows:

| Panel | Description |
|-------|-------------|
| **Price + SMA50/200/EMA20 + Target** | 6-month daily price with SMA50 (amber), SMA200 (purple), EMA20 (blue dashed), Golden/Death Cross marker, and projected target |
| **MACD (12, 26, 9)** | Histogram (green/red bars), MACD line, and signal line with current crossover status in the title |
| **RSI (14)** | RSI line with overbought (>70) and oversold (<30) fill zones; current value and zone label in title |
| **Stochastic (14, 3)** | %K (fast) and %D (signal) lines with overbought (>80) and oversold (<20) fill zones; current values in title |
| **Confidence & Risk** | Arc gauge showing confidence %, risk pill, and direction label |
| **Technical Signal Factors** | Horizontal bar chart of the indicator signals that drove the prediction direction |

## Technical Indicators

The prediction engine computes these trend-following indicators on 1 year of daily closes from Yahoo Finance:

| Indicator | Detail |
|-----------|--------|
| **SMA50** | 50-day simple moving average |
| **SMA200** | 200-day simple moving average |
| **EMA20** | 20-day exponential moving average |
| **Golden Cross** | SMA50 crosses above SMA200 — bullish long-term signal (+2 pts) |
| **Death Cross** | SMA50 crosses below SMA200 — bearish long-term signal (+2 pts) |
| **MACD (12, 26, 9)** | MACD line (EMA12 − EMA26), signal line (EMA9 of MACD), histogram |
| **MACD crossover** | MACD line crossing above/below signal line (+2 pts) |
| **Price vs SMA50/200** | Whether price trades above or below each MA (+1 pt each) |
| **RSI (14)** | <30 oversold = bullish (+2 pts), >70 overbought = bearish (+2 pts), above/below 50 midline (+1 pt) |
| **Stochastic (14, 3)** | %K/%D crossover (+1 pt); %K <20 oversold = bullish (+1 pt), >80 overbought = bearish (+1 pt) |

Direction (`bullish` / `bearish` / `neutral`) and confidence are derived by scoring these signals — no random guessing.

## Stock Prediction Tool Reference

The **Stock Prediction** tool is defined as an Anthropic tool-use schema. Claude calls it automatically when asked for a stock prediction.

**Input parameters:**

| Parameter   | Type   | Required | Description                                       |
|-------------|--------|----------|---------------------------------------------------|
| `ticker`    | string | Yes      | Stock ticker symbol (e.g., `AAPL`, `TSLA`)        |
| `timeframe` | string | No       | One of `1d`, `1w`, `1m`, `3m`, `6m` — defaults to `1w`  |

**Output fields:**

| Field           | Description                                                     |
|-----------------|-----------------------------------------------------------------|
| `ticker`        | Uppercased ticker symbol                                        |
| `timeframe`     | Requested prediction window                                     |
| `direction`     | `bullish`, `bearish`, or `neutral` — derived from indicator scores |
| `confidence`    | Confidence score scaled by signal strength                      |
| `current_price` | Live price fetched from Yahoo Finance                           |
| `price_target`  | Projected price at the end of the timeframe                     |
| `target_date`   | ISO date when the target should be reached                      |
| `key_factors`   | Up to 5 indicator signals that drove the direction              |
| `risk_level`    | `low`, `medium`, or `high`                                      |

## Project Structure

```
stock-prediction/
├── stock_predictor.py   # Main application
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── results/             # Output from each run (auto-created)
│   └── YYYYMMDD_HHMMSS/
│       ├── predictions.md
│       └── charts/
└── .venv/               # Virtual environment (not committed)
```

## Prompt Caching

The system prompt is marked with `cache_control: {type: "ephemeral"}`. On repeated calls within a 5-minute window, Anthropic serves the cached prefix at ~10% of the normal input token cost. Cache hits appear in the output as:

```
Cache read: 312 tokens
```

> Note: Caching requires a minimum prefix length (~2048 tokens for Sonnet). The system prompt in this demo is intentionally short for clarity; extend it with more detailed instructions to trigger caching in production.

## Disclaimer

Predictions are for **demonstration purposes only**. Current prices and indicator data are fetched live from Yahoo Finance, and direction/confidence are derived from real technical signals — but technical analysis does not guarantee future performance. This tool should not be used to make investment decisions.
