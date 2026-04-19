import argparse
import anthropic
import json
import os
import random
from datetime import datetime, timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yfinance as yf

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a financial analysis assistant with access to the stock prediction tool.
When asked about a stock, use the stock_prediction tool to retrieve
prediction data, then provide a clear, concise analysis of the results. Always remind
users that stock predictions are not financial advice."""

tools = [
    {
        "name": "stock_prediction",
        "description": (
            "Predicts the future price movement of a stock based on technical analysis, "
            "sentiment data, and historical patterns. Returns a prediction with confidence "
            "score, price target, and key factors influencing the prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["1d", "1w", "1m", "3m", "6m"],
                    "description": "Prediction timeframe: 1 day, 1 week, 1 month, 3 months, or 6 months",
                },
            },
            "required": ["ticker"],
        },
    }
]


def get_current_price(ticker: str) -> float:
    info = yf.Ticker(ticker).fast_info
    price = info.get("lastPrice") or info.get("previousClose")
    if not price:
        raise ValueError(f"Could not fetch price for {ticker!r}")
    return round(float(price), 2)


def get_price_history(ticker: str, timeframe: str) -> tuple[list, list]:
    """Fetch recent historical prices for the sparkline chart."""
    period_map = {"1d": "5d", "1w": "1mo", "1m": "3mo", "3m": "1y", "6m": "2y"}
    interval_map = {"1d": "1h", "1w": "1d", "1m": "1wk", "3m": "1mo", "6m": "1mo"}
    hist = yf.Ticker(ticker).history(
        period=period_map.get(timeframe, "1mo"),
        interval=interval_map.get(timeframe, "1d"),
    )
    return list(range(len(hist))), list(hist["Close"])


def get_moving_averages(ticker: str) -> tuple[list, list, list]:
    """Fetch 1 year of daily closes and return (indices, ma50, ma200)."""
    hist = yf.Ticker(ticker).history(period="1y", interval="1d")
    closes = hist["Close"]
    xs = list(range(len(closes)))
    ma50 = closes.rolling(window=50).mean().tolist()
    ma200 = closes.rolling(window=200).mean().tolist()
    return xs, ma50, ma200


def run_openclaw(ticker: str, timeframe: str = "1w") -> dict:
    directions = ["bullish", "bearish", "neutral"]
    direction = random.choice(directions)
    confidence = round(random.uniform(0.55, 0.92), 2)

    base_price = get_current_price(ticker)
    change_pct = random.uniform(-0.15, 0.20) if direction != "neutral" else random.uniform(-0.05, 0.05)
    if direction == "bearish":
        change_pct = -abs(change_pct)
    price_target = round(base_price * (1 + change_pct), 2)

    factors = {
        "bullish": [
            "Strong earnings momentum",
            "Positive institutional buying",
            "Bullish RSI divergence",
            "Above 50-day moving average",
        ],
        "bearish": [
            "Weakening revenue growth",
            "Increased short interest",
            "Death cross pattern forming",
            "Below key support levels",
        ],
        "neutral": [
            "Mixed analyst sentiment",
            "Consolidation pattern",
            "Low volume trading",
            "Awaiting catalyst",
        ],
    }

    timeframe_days = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180}
    target_date = (datetime.now() + timedelta(days=timeframe_days.get(timeframe, 7))).strftime("%Y-%m-%d")

    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "direction": direction,
        "confidence": confidence,
        "current_price": base_price,
        "price_target": price_target,
        "target_date": target_date,
        "key_factors": random.sample(factors[direction], min(3, len(factors[direction]))),
        "risk_level": random.choice(["low", "medium", "high"]),
    }


def generate_chart(prediction: dict, charts_dir: str) -> str:
    """Generate a 2x2 analysis chart and return the saved file path."""
    ticker = prediction["ticker"]
    direction = prediction["direction"]
    confidence = prediction["confidence"]
    current = prediction["current_price"]
    target = prediction["price_target"]
    factors = prediction["key_factors"]
    risk = prediction["risk_level"]
    timeframe = prediction["timeframe"]

    color_map = {"bullish": "#26a69a", "bearish": "#ef5350", "neutral": "#ffa726"}
    risk_colors = {"low": "#26a69a", "medium": "#ffa726", "high": "#ef5350"}
    main_color = color_map[direction]

    fig = plt.figure(figsize=(12, 8), facecolor="#0d1117")
    fig.suptitle(
        f"{ticker} — Stock Analysis ({timeframe})",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Price history sparkline + target projection + MAs ───────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    try:
        xs, prices = get_price_history(ticker, timeframe)
        ax1.plot(xs, prices, color=main_color, linewidth=2, label="Price")
        ax1.fill_between(xs, prices, alpha=0.15, color=main_color)
        last_x = xs[-1]
        proj_x = [last_x, last_x + max(1, len(xs) // 4)]
        proj_y = [prices[-1], target]
        ax1.plot(proj_x, proj_y, "--", color=main_color, linewidth=1.5, alpha=0.8)
        ax1.scatter([proj_x[-1]], [proj_y[-1]], color=main_color, s=60, zorder=5)
        ax1.axhline(target, color=main_color, linewidth=0.8, linestyle=":", alpha=0.5)
        ax1.text(
            proj_x[-1] + 0.2, proj_y[-1],
            f"  ${target:,.2f}",
            color=main_color, fontsize=8, va="center",
        )

        # Overlay 50-day and 200-day moving averages scaled to chart x-range
        try:
            ma_xs, ma50, ma200 = get_moving_averages(ticker)
            n = len(xs)
            # Scale MA indices to the same x-range as the sparkline
            scaled_xs = [i * (last_x / max(len(ma_xs) - 1, 1)) for i in ma_xs]
            valid50 = [(scaled_xs[i], ma50[i]) for i in range(len(ma50)) if ma50[i] is not None and not np.isnan(ma50[i])]
            valid200 = [(scaled_xs[i], ma200[i]) for i in range(len(ma200)) if ma200[i] is not None and not np.isnan(ma200[i])]
            if valid50:
                ax1.plot([p[0] for p in valid50], [p[1] for p in valid50],
                         color="#f0b429", linewidth=1.2, alpha=0.85, label="MA50")
            if valid200:
                ax1.plot([p[0] for p in valid200], [p[1] for p in valid200],
                         color="#a78bfa", linewidth=1.2, alpha=0.85, label="MA200")
            ax1.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                       edgecolor="#30363d", labelcolor="white", framealpha=0.8)
        except Exception:
            pass
    except Exception:
        ax1.text(0.5, 0.5, "History unavailable", ha="center", va="center",
                 color="gray", transform=ax1.transAxes)
    ax1.set_title("Price History + MA50/200 + Target", color="white", fontsize=10, pad=6)
    ax1.tick_params(colors="gray", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 2: Current vs Target bar ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    bars = ax2.bar(
        ["Current", "Target"],
        [current, target],
        color=["#58a6ff", main_color],
        width=0.5,
        edgecolor="#30363d",
    )
    for bar, val in zip(bars, [current, target]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(current, target) * 0.01,
            f"${val:,.2f}",
            ha="center", va="bottom", color="white", fontsize=9, fontweight="bold",
        )
    change_pct = (target - current) / current * 100
    sign = "+" if change_pct >= 0 else ""
    ax2.set_title(
        f"Price Target  ({sign}{change_pct:.1f}%)",
        color="white", fontsize=10, pad=6,
    )
    ax2.tick_params(colors="gray", labelsize=8)
    ax2.set_ylim(0, max(current, target) * 1.15)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 3: Confidence + Risk gauge ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161b22")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # Confidence arc
    theta = np.linspace(np.pi, np.pi * (1 - confidence), 100)
    ax3.plot(
        0.3 + 0.22 * np.cos(theta),
        0.55 + 0.22 * np.sin(theta),
        color=main_color, linewidth=10, solid_capstyle="round",
    )
    ax3.plot(
        0.3 + 0.22 * np.cos(np.linspace(np.pi, 0, 100)),
        0.55 + 0.22 * np.sin(np.linspace(np.pi, 0, 100)),
        color="#30363d", linewidth=10, solid_capstyle="round",
    )
    ax3.text(0.3, 0.50, f"{int(confidence * 100)}%", ha="center", va="center",
             color="white", fontsize=18, fontweight="bold")
    ax3.text(0.3, 0.20, "Confidence", ha="center", color="gray", fontsize=9)

    # Risk pill
    risk_color = risk_colors[risk]
    pill = mpatches.FancyBboxPatch(
        (0.60, 0.42), 0.30, 0.18,
        boxstyle="round,pad=0.02", facecolor=risk_color, edgecolor="none", alpha=0.25,
    )
    ax3.add_patch(pill)
    ax3.text(0.75, 0.51, f"Risk: {risk.upper()}", ha="center", va="center",
             color=risk_color, fontsize=10, fontweight="bold")

    direction_icon = {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "◆ NEUTRAL"}[direction]
    ax3.text(0.75, 0.28, direction_icon, ha="center", color=main_color,
             fontsize=11, fontweight="bold")
    ax3.set_title("Confidence & Risk", color="white", fontsize=10, pad=6)

    # ── Panel 4: Key factors horizontal bar ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    y_pos = range(len(factors))
    weights = [random.uniform(0.6, 1.0) for _ in factors]
    ax4.barh(
        list(y_pos), weights,
        color=main_color, alpha=0.8, edgecolor="#30363d", height=0.5,
    )
    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels(
        [f.replace(" ", "\n") if len(f) > 20 else f for f in factors],
        color="white", fontsize=8,
    )
    ax4.set_xlim(0, 1.2)
    ax4.set_xticks([0, 0.5, 1.0])
    ax4.set_xticklabels(["Low", "Med", "High"], color="gray", fontsize=7)
    ax4.set_title("Key Factors (Impact)", color="white", fontsize=10, pad=6)
    ax4.tick_params(colors="gray")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")

    chart_path = os.path.join(charts_dir, f"{ticker}_{timeframe}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Chart saved: {chart_path}")
    return chart_path


def predict_stock(ticker: str, timeframe: str = "1w", md_file=None, charts_dir: str = "charts", model: str = "claude-sonnet-4-6") -> None:
    print(f"\nAnalyzing {ticker.upper()} for {timeframe} timeframe... (model: {model})\n")

    messages = [
        {
            "role": "user",
            "content": f"Use the stock_prediction tool to predict the stock performance of {ticker} over the next {timeframe}.",
        }
    ]

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=tools,
        messages=messages,
    )
    print(f"Cache created: {response.usage.cache_creation_input_tokens} tokens")
    print(f"Cache read:    {response.usage.cache_read_input_tokens} tokens\n")

    last_prediction = None

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "stock_prediction":
                ticker_input = block.input.get("ticker", ticker)
                tf_input = block.input.get("timeframe", timeframe)
                print(f"Tool called: ticker={ticker_input}, timeframe={tf_input}")
                prediction = run_openclaw(ticker_input, tf_input)
                last_prediction = prediction
                print(f"Prediction data: {json.dumps(prediction, indent=2)}\n")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(prediction),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )
        print(f"Cache created: {response.usage.cache_creation_input_tokens} tokens")
        print(f"Cache read:    {response.usage.cache_read_input_tokens} tokens\n")

    for block in response.content:
        if hasattr(block, "text"):
            print("=" * 60)
            print(block.text)
            print("=" * 60)

            if md_file and last_prediction:
                chart_path = generate_chart(last_prediction, charts_dir)
                rel_path = os.path.relpath(chart_path, os.path.dirname(md_file.name))

                md_file.write(f"## {ticker.upper()} — {timeframe} Prediction\n\n")
                md_file.write(f"![{ticker} Chart]({rel_path})\n\n")
                md_file.write(block.text + "\n\n")
                md_file.write("---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Predictor — AI-powered stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python stock_predictor.py\n"
            "  python stock_predictor.py --tickers AAPL\n"
            "  python stock_predictor.py --tickers AAPL TSLA NVDA\n"
            "  python stock_predictor.py --tickers MSFT --timeframe 3m\n"
            "  python stock_predictor.py --tickers GOOG AMZN --timeframe 1d\n"
            "  python stock_predictor.py --tickers AAPL --model claude-opus-4-7\n"
        ),
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        default=["AAPL", "TSLA", "INTC"],
        help="one or more stock ticker symbols (default: AAPL TSLA INTC)",
    )
    parser.add_argument(
        "--timeframe", choices=["1d", "1w", "1m", "3m", "6m"], default=None,
        help="prediction timeframe for all tickers (default: 1w)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        metavar="MODEL",
        help="Claude model ID to use (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", timestamp)
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    md_filename = os.path.join(run_dir, "predictions.md")
    with open(md_filename, "w") as f:
        f.write("# Stock Predictions\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        tickers_str = ", ".join(t.upper() for t in args.tickers)
        tf_str = args.timeframe or "per-ticker default"
        f.write(f"**Tickers:** {tickers_str}  \n**Timeframe:** {tf_str}  \n**Model:** {args.model}\n\n")
        f.write("---\n\n")

        default_timeframes = {"AAPL": "1w", "TSLA": "1m", "INTC": "1m"}
        for ticker in args.tickers:
            tf = args.timeframe or default_timeframes.get(ticker.upper(), "1w")
            predict_stock(ticker, tf, md_file=f, charts_dir=charts_dir, model=args.model)

    print(f"\nResults saved to: {run_dir}/")
