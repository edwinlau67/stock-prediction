# Stock Predictor

A Python application that uses the [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) and Claude's tool use feature to predict stock performance. Claude calls the **Stock Prediction** tool to retrieve real-time price data and generate predictions, then delivers a structured analysis with price targets, confidence scores, and key market factors — saved as a Markdown report with analysis charts.

## How It Works

```
CLI args → Claude → Stock Prediction tool call → Live price (yfinance) + prediction data → Claude analysis → Markdown report + charts
```

1. You specify one or more tickers and a timeframe via CLI arguments
2. Claude invokes the **Stock Prediction** tool with the ticker and timeframe
3. Stock Prediction fetches the **real current price** from Yahoo Finance and generates prediction data
4. Claude analyzes the data and writes a formatted Markdown report
5. A 4-panel analysis chart (PNG) is generated per ticker
6. All output is saved into a timestamped folder under `results/`

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

Each PNG chart contains 4 panels:

| Panel | Description |
|-------|-------------|
| **Price History + MA50/200 + Target** | Historical price sparkline with 50-day (amber) and 200-day (purple) moving averages and projected target line |
| **Price Target** | Current vs target price bar chart with % change |
| **Confidence & Risk** | Arc gauge showing confidence %, risk pill, and direction label |
| **Key Factors (Impact)** | Horizontal bar chart of the factors driving the prediction |

## Stock Prediction Tool Reference

The **Stock Prediction** tool is defined as an Anthropic tool-use schema. Claude calls it automatically when asked for a stock prediction.

**Input parameters:**

| Parameter   | Type   | Required | Description                                       |
|-------------|--------|----------|---------------------------------------------------|
| `ticker`    | string | Yes      | Stock ticker symbol (e.g., `AAPL`, `TSLA`)        |
| `timeframe` | string | No       | One of `1d`, `1w`, `1m`, `3m`, `6m` — defaults to `1w`  |

**Output fields:**

| Field           | Description                                           |
|-----------------|-------------------------------------------------------|
| `ticker`        | Uppercased ticker symbol                              |
| `timeframe`     | Requested prediction window                           |
| `direction`     | `bullish`, `bearish`, or `neutral`                    |
| `confidence`    | Confidence score between 0.55 and 0.92                |
| `current_price` | Live price fetched from Yahoo Finance                 |
| `price_target`  | Predicted price at the end of the timeframe           |
| `target_date`   | ISO date when the target should be reached            |
| `key_factors`   | Up to 3 factors driving the prediction                |
| `risk_level`    | `low`, `medium`, or `high`                            |

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

Predictions are **simulated** and for demonstration purposes only. While current prices are fetched live from Yahoo Finance, the prediction direction, confidence, and key factors are algorithmically generated and do not reflect real financial analysis. This tool should not be used to make investment decisions.
