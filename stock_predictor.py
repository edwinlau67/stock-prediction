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



def _valid(v) -> bool:
    try:
        return v is not None and not np.isnan(v)
    except (TypeError, ValueError):
        return False


def get_technical_indicators(ticker: str) -> dict:
    """Fetch 1 year of daily closes and compute trend-following indicators."""
    hist = yf.Ticker(ticker).history(period="1y", interval="1d")
    closes = hist["Close"]

    sma50 = closes.rolling(window=50).mean()
    sma200 = closes.rolling(window=200).mean()
    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    s50 = sma50.tolist()
    s200 = sma200.tolist()
    ml = macd_line.tolist()
    sl = signal_line.tolist()

    # Most recent Golden Cross / Death Cross over the full year
    cross_signal = None
    cross_idx = None
    for i in range(1, len(s50)):
        if not all(_valid(v) for v in [s50[i-1], s200[i-1], s50[i], s200[i]]):
            continue
        if s50[i-1] < s200[i-1] and s50[i] >= s200[i]:
            cross_signal, cross_idx = "golden", i
        elif s50[i-1] > s200[i-1] and s50[i] <= s200[i]:
            cross_signal, cross_idx = "death", i

    # Most recent MACD line / signal-line crossover
    macd_crossover = None
    for i in range(1, len(ml)):
        if not all(_valid(v) for v in [ml[i-1], sl[i-1], ml[i], sl[i]]):
            continue
        if ml[i-1] < sl[i-1] and ml[i] >= sl[i]:
            macd_crossover = "bullish"
        elif ml[i-1] > sl[i-1] and ml[i] <= sl[i]:
            macd_crossover = "bearish"

    return {
        "xs": list(range(len(closes))),
        "closes": closes.tolist(),
        "sma50": s50,
        "sma200": s200,
        "ema20": ema20.tolist(),
        "macd_line": ml,
        "signal_line": sl,
        "histogram": histogram.tolist(),
        "cross_signal": cross_signal,
        "cross_idx": cross_idx,
        "macd_crossover": macd_crossover,
    }


def run_prediction(ticker: str, timeframe: str = "1w") -> dict:
    base_price = get_current_price(ticker)

    try:
        tech = get_technical_indicators(ticker)
    except Exception:
        tech = None

    bullish_score = 0
    bearish_score = 0
    key_factors: list[str] = []

    if tech:
        last_sma50 = next((v for v in reversed(tech["sma50"]) if _valid(v)), None)
        last_sma200 = next((v for v in reversed(tech["sma200"]) if _valid(v)), None)
        last_macd = next((v for v in reversed(tech["macd_line"]) if _valid(v)), None)
        last_signal = next((v for v in reversed(tech["signal_line"]) if _valid(v)), None)

        # Golden / Death Cross
        if tech["cross_signal"] == "golden":
            bullish_score += 2
            key_factors.append("Golden Cross: SMA50 crossed above SMA200")
        elif tech["cross_signal"] == "death":
            bearish_score += 2
            key_factors.append("Death Cross: SMA50 crossed below SMA200")

        # MACD crossover signal
        if tech["macd_crossover"] == "bullish":
            bullish_score += 2
            key_factors.append("MACD bullish crossover — momentum building")
        elif tech["macd_crossover"] == "bearish":
            bearish_score += 2
            key_factors.append("MACD bearish crossover — momentum weakening")

        # Current MACD vs signal line
        if last_macd is not None and last_signal is not None:
            if last_macd > last_signal:
                bullish_score += 1
                key_factors.append(f"MACD ({last_macd:.3f}) above signal ({last_signal:.3f})")
            else:
                bearish_score += 1
                key_factors.append(f"MACD ({last_macd:.3f}) below signal ({last_signal:.3f})")

        # Price vs SMA50 / SMA200
        if last_sma50:
            if base_price > last_sma50:
                bullish_score += 1
                key_factors.append(f"Price above SMA50 (${last_sma50:.2f})")
            else:
                bearish_score += 1
                key_factors.append(f"Price below SMA50 (${last_sma50:.2f})")
        if last_sma200:
            if base_price > last_sma200:
                bullish_score += 1
                key_factors.append(f"Price above SMA200 (${last_sma200:.2f})")
            else:
                bearish_score += 1
                key_factors.append(f"Price below SMA200 (${last_sma200:.2f})")

    if bullish_score > bearish_score:
        direction = "bullish"
    elif bearish_score > bullish_score:
        direction = "bearish"
    else:
        direction = "neutral"

    signal_gap = abs(bullish_score - bearish_score)
    confidence = round(min(0.95, 0.52 + signal_gap * 0.05 + random.uniform(0, 0.08)), 2)

    if direction == "bullish":
        change_pct = random.uniform(0.02, 0.15)
    elif direction == "bearish":
        change_pct = random.uniform(-0.15, -0.02)
    else:
        change_pct = random.uniform(-0.05, 0.05)
    price_target = round(base_price * (1 + change_pct), 2)

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
        "key_factors": key_factors[:5],
        "risk_level": random.choice(["low", "medium", "high"]),
        "technical": tech,
    }


def generate_chart(prediction: dict, charts_dir: str) -> str:
    """Generate a 3-row technical analysis chart and return the saved file path."""
    ticker = prediction["ticker"]
    direction = prediction["direction"]
    confidence = prediction["confidence"]
    current = prediction["current_price"]
    target = prediction["price_target"]
    factors = prediction["key_factors"]
    risk = prediction["risk_level"]
    timeframe = prediction["timeframe"]
    tech = prediction.get("technical") or {}

    color_map = {"bullish": "#26a69a", "bearish": "#ef5350", "neutral": "#ffa726"}
    risk_colors = {"low": "#26a69a", "medium": "#ffa726", "high": "#ef5350"}
    main_color = color_map[direction]

    fig = plt.figure(figsize=(14, 11), facecolor="#0d1117")
    change_pct = (target - current) / current * 100
    sign = "+" if change_pct >= 0 else ""
    fig.suptitle(
        f"{ticker} — Technical Analysis ({timeframe})  ·  Target ${target:,.2f} ({sign}{change_pct:.1f}%)",
        fontsize=15, fontweight="bold", color="white", y=0.99,
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[2.5, 1.5, 1.5],
        hspace=0.52, wspace=0.35,
    )

    # ── Panel 1: Price + SMA50/200 + EMA20 + Golden/Death Cross (full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22")

    if tech:
        n_all = len(tech["xs"])
        display_n = min(126, n_all)       # last ~6 months of trading days
        offset = n_all - display_n
        xs = list(range(display_n))

        closes = tech["closes"][-display_n:]
        sma50  = tech["sma50"][-display_n:]
        sma200 = tech["sma200"][-display_n:]
        ema20  = tech["ema20"][-display_n:]

        ax1.plot(xs, closes, color=main_color, linewidth=1.8, label="Price", zorder=3)
        ax1.fill_between(xs, closes, alpha=0.10, color=main_color)

        v50 = [(xs[i], sma50[i]) for i in range(len(sma50)) if _valid(sma50[i])]
        if v50:
            ax1.plot([p[0] for p in v50], [p[1] for p in v50],
                     color="#f0b429", linewidth=1.4, alpha=0.9, label="SMA50")

        v200 = [(xs[i], sma200[i]) for i in range(len(sma200)) if _valid(sma200[i])]
        if v200:
            ax1.plot([p[0] for p in v200], [p[1] for p in v200],
                     color="#a78bfa", linewidth=1.4, alpha=0.9, label="SMA200")

        ve20 = [(xs[i], ema20[i]) for i in range(len(ema20)) if _valid(ema20[i])]
        if ve20:
            ax1.plot([p[0] for p in ve20], [p[1] for p in ve20],
                     color="#38bdf8", linewidth=1.1, linestyle="--", alpha=0.85, label="EMA20")

        # Golden / Death Cross marker (only if within display window)
        cross_idx = tech.get("cross_idx")
        if cross_idx is not None:
            disp_idx = cross_idx - offset
            if 0 <= disp_idx < display_n:
                cx_y = closes[disp_idx]
                is_golden = tech["cross_signal"] == "golden"
                cx_color = "#ffd700" if is_golden else "#ff4444"
                cx_label = "Golden Cross ★" if is_golden else "Death Cross ✕"
                ax1.scatter([disp_idx], [cx_y], color=cx_color, s=180, zorder=6,
                            marker="*" if is_golden else "X", label=cx_label)

        # Target projection arrow
        last_x = xs[-1]
        proj_len = max(4, display_n // 8)
        proj_x = [last_x, last_x + proj_len]
        proj_y = [closes[-1], target]
        ax1.plot(proj_x, proj_y, "--", color=main_color, linewidth=1.5, alpha=0.8)
        ax1.scatter([proj_x[-1]], [proj_y[-1]], color=main_color, s=70, zorder=5)
        ax1.text(proj_x[-1] + 0.5, proj_y[-1], f"  ${target:,.2f}",
                 color=main_color, fontsize=8, va="center")

        ax1.legend(fontsize=7, loc="upper left", facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="white", framealpha=0.8, ncol=4)
    else:
        ax1.text(0.5, 0.5, "Technical data unavailable", ha="center", va="center",
                 color="gray", transform=ax1.transAxes)

    ax1.set_title("Price  +  SMA50 / SMA200 / EMA20  +  Target Projection",
                  color="white", fontsize=10, pad=6)
    ax1.tick_params(colors="gray", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 2: MACD (12, 26, 9) — full width ────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor("#161b22")

    if tech:
        n_all = len(tech["xs"])
        display_n = min(126, n_all)
        xs_m = list(range(display_n))
        ml = tech["macd_line"][-display_n:]
        sl = tech["signal_line"][-display_n:]
        hl = tech["histogram"][-display_n:]

        # Histogram bars (green = positive, red = negative)
        valid_hi = [(xs_m[i], hl[i]) for i in range(len(hl)) if _valid(hl[i])]
        if valid_hi:
            bar_xs = [p[0] for p in valid_hi]
            bar_ys = [p[1] for p in valid_hi]
            bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in bar_ys]
            ax2.bar(bar_xs, bar_ys, color=bar_colors, alpha=0.5, width=1.0, label="Histogram")

        valid_ml = [(xs_m[i], ml[i]) for i in range(len(ml)) if _valid(ml[i])]
        if valid_ml:
            ax2.plot([p[0] for p in valid_ml], [p[1] for p in valid_ml],
                     color="#38bdf8", linewidth=1.5, label="MACD (12−26)")

        valid_sl = [(xs_m[i], sl[i]) for i in range(len(sl)) if _valid(sl[i])]
        if valid_sl:
            ax2.plot([p[0] for p in valid_sl], [p[1] for p in valid_sl],
                     color="#ffa726", linewidth=1.2, label="Signal (9)")

        ax2.axhline(0, color="#30363d", linewidth=0.8)

        last_m = next((v for v in reversed(ml) if _valid(v)), None)
        last_s = next((v for v in reversed(sl) if _valid(v)), None)
        if last_m is not None and last_s is not None:
            trend = "▲ Bullish" if last_m > last_s else "▼ Bearish"
            status = f"{trend}   MACD {last_m:+.3f}   Signal {last_s:+.3f}"
        else:
            status = ""
        ax2.set_title(f"MACD (12, 26, 9)   {status}", color="white", fontsize=10, pad=6)
        ax2.legend(fontsize=7, loc="upper left", facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="white", framealpha=0.8)
    else:
        ax2.text(0.5, 0.5, "MACD unavailable", ha="center", va="center",
                 color="gray", transform=ax2.transAxes)
        ax2.set_title("MACD (12, 26, 9)", color="white", fontsize=10, pad=6)

    ax2.tick_params(colors="gray", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 3: Confidence + Risk gauge ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor("#161b22")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    theta = np.linspace(np.pi, np.pi * (1 - confidence), 100)
    ax3.plot(
        0.3 + 0.22 * np.cos(np.linspace(np.pi, 0, 100)),
        0.55 + 0.22 * np.sin(np.linspace(np.pi, 0, 100)),
        color="#30363d", linewidth=10, solid_capstyle="round",
    )
    ax3.plot(
        0.3 + 0.22 * np.cos(theta),
        0.55 + 0.22 * np.sin(theta),
        color=main_color, linewidth=10, solid_capstyle="round",
    )
    ax3.text(0.3, 0.50, f"{int(confidence * 100)}%", ha="center", va="center",
             color="white", fontsize=18, fontweight="bold")
    ax3.text(0.3, 0.20, "Confidence", ha="center", color="gray", fontsize=9)

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

    # ── Panel 4: Technical signal factors ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor("#161b22")
    if factors:
        y_pos = range(len(factors))
        weights = [random.uniform(0.6, 1.0) for _ in factors]
        ax4.barh(list(y_pos), weights, color=main_color, alpha=0.8,
                 edgecolor="#30363d", height=0.5)
        ax4.set_yticks(list(y_pos))
        ax4.set_yticklabels(
            [f[:30] + "…" if len(f) > 30 else f for f in factors],
            color="white", fontsize=7,
        )
        ax4.set_xlim(0, 1.2)
        ax4.set_xticks([0, 0.5, 1.0])
        ax4.set_xticklabels(["Low", "Med", "High"], color="gray", fontsize=7)
    else:
        ax4.text(0.5, 0.5, "No signals", ha="center", va="center",
                 color="gray", transform=ax4.transAxes)
    ax4.set_title("Technical Signal Factors", color="white", fontsize=10, pad=6)
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
                prediction = run_prediction(ticker_input, tf_input)
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
