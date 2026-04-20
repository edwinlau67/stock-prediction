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
                    "enum": ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "2y", "5y"],
                    "description": "Prediction timeframe: 1d, 1w, 1m, 3m, 6m, ytd, 1y, 2y, or 5y",
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


def get_fundamental_indicators(ticker: str) -> dict:
    """Fetch key fundamental metrics from Yahoo Finance."""
    info = yf.Ticker(ticker).info

    def _get(key):
        v = info.get(key)
        if v is None or v == "N/A":
            return None
        try:
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    return {
        "trailing_pe":      _get("trailingPE"),
        "forward_pe":       _get("forwardPE"),
        "price_to_book":    _get("priceToBook"),
        "price_to_sales":   _get("priceToSalesTrailing12Months"),
        "ev_ebitda":        _get("enterpriseToEbitda"),
        "peg_ratio":        _get("pegRatio"),
        "trailing_eps":     _get("trailingEps"),
        "forward_eps":      _get("forwardEps"),
        "earnings_growth":  _get("earningsGrowth"),   # decimal, e.g. 0.15
        "revenue_growth":   _get("revenueGrowth"),    # decimal
        "gross_margin":     _get("grossMargins"),      # decimal
        "operating_margin": _get("operatingMargins"),  # decimal
        "net_margin":       _get("profitMargins"),     # decimal
        "roe":              _get("returnOnEquity"),    # decimal
        "roa":              _get("returnOnAssets"),    # decimal
        "debt_to_equity":   _get("debtToEquity"),     # yfinance: percentage * 100
        "current_ratio":    _get("currentRatio"),
        "dividend_yield":   _get("dividendYield"),    # decimal
        "market_cap":       _get("marketCap"),
        "short_ratio":      _get("shortRatio"),
    }


def get_technical_indicators(ticker: str) -> dict:
    """Fetch 1 year of daily OHLC and compute trend and momentum indicators."""
    hist = yf.Ticker(ticker).history(period="1y", interval="1d")
    closes = hist["Close"]
    highs  = hist["High"]
    lows   = hist["Low"]
    volume = hist["Volume"]

    # ── Trend indicators ──────────────────────────────────────────────────
    sma50 = closes.rolling(window=50).mean()
    sma200 = closes.rolling(window=200).mean()
    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    # ── Momentum indicators ───────────────────────────────────────────────
    # RSI (14-period)
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Stochastic Oscillator — %K (14-period), %D (3-period SMA of %K)
    low14  = lows.rolling(window=14).min()
    high14 = highs.rolling(window=14).max()
    stoch_k = 100 * (closes - low14) / (high14 - low14)
    stoch_d = stoch_k.rolling(window=3).mean()

    s50 = sma50.tolist()
    s200 = sma200.tolist()
    ml = macd_line.tolist()
    sl = signal_line.tolist()
    sk = stoch_k.tolist()
    sd = stoch_d.tolist()

    # Most recent Golden Cross / Death Cross
    cross_signal = None
    cross_idx = None
    for i in range(1, len(s50)):
        if not all(_valid(v) for v in [s50[i-1], s200[i-1], s50[i], s200[i]]):
            continue
        if s50[i-1] < s200[i-1] and s50[i] >= s200[i]:
            cross_signal, cross_idx = "golden", i
        elif s50[i-1] > s200[i-1] and s50[i] <= s200[i]:
            cross_signal, cross_idx = "death", i

    # Most recent MACD crossover
    macd_crossover = None
    for i in range(1, len(ml)):
        if not all(_valid(v) for v in [ml[i-1], sl[i-1], ml[i], sl[i]]):
            continue
        if ml[i-1] < sl[i-1] and ml[i] >= sl[i]:
            macd_crossover = "bullish"
        elif ml[i-1] > sl[i-1] and ml[i] <= sl[i]:
            macd_crossover = "bearish"

    # Most recent Stochastic %K / %D crossover
    stoch_crossover = None
    for i in range(1, len(sk)):
        if not all(_valid(v) for v in [sk[i-1], sd[i-1], sk[i], sd[i]]):
            continue
        if sk[i-1] < sd[i-1] and sk[i] >= sd[i]:
            stoch_crossover = "bullish"
        elif sk[i-1] > sd[i-1] and sk[i] <= sd[i]:
            stoch_crossover = "bearish"

    # ── Volatility indicators ─────────────────────────────────────────────
    # Bollinger Bands (SMA20 ± 2 standard deviations)
    bb_mid   = closes.rolling(window=20).mean()
    bb_std   = closes.rolling(window=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    last_close = closes.iloc[-1]
    last_upper = bb_upper.iloc[-1]
    last_lower = bb_lower.iloc[-1]
    if _valid(last_upper) and _valid(last_lower):
        if last_close > last_upper:
            bb_signal = "above_upper"
        elif last_close < last_lower:
            bb_signal = "below_lower"
        else:
            bb_signal = "within"
    else:
        bb_signal = None

    # ATR (14-period, Wilder's smoothing via EWM)
    prev_close = closes.shift(1)
    hl  = highs - lows
    hpc = (highs - prev_close).abs()
    lpc = (lows  - prev_close).abs()
    tr  = hl.where(hl >= hpc, hpc).where(hl.where(hl >= hpc, hpc) >= lpc, lpc)
    atr = tr.ewm(span=14, adjust=False).mean()
    atr_mean = atr.rolling(window=20).mean()

    atr_vals = atr.tolist()
    atr_mean_vals = atr_mean.tolist()
    last_atr  = next((v for v in reversed(atr_vals) if _valid(v)), None)
    last_atrm = next((v for v in reversed(atr_mean_vals) if _valid(v)), None)
    if last_atr and last_atrm and last_atrm > 0:
        atr_ratio = last_atr / last_atrm
        atr_level = "high" if atr_ratio > 1.3 else ("low" if atr_ratio < 0.8 else "medium")
    else:
        atr_ratio, atr_level = 1.0, "medium"

    # ── Volume indicators ─────────────────────────────────────────────────
    # OBV: add volume on up days, subtract on down days
    obv = (np.sign(closes.diff()) * volume).fillna(0).cumsum()

    # Volume spikes: day's volume > 20-day mean + 2 std deviations
    vol_mean = volume.rolling(window=20).mean()
    vol_std  = volume.rolling(window=20).std()
    vol_spike = volume > (vol_mean + 2 * vol_std)

    # OBV short-term trend: slope over last 10 valid bars
    obv_vals = obv.tolist()
    recent_obv = [v for v in obv_vals[-10:] if _valid(v)]
    obv_trend = ("rising" if len(recent_obv) >= 2 and recent_obv[-1] > recent_obv[0]
                 else "falling" if len(recent_obv) >= 2 else None)

    # Most recent volume spike direction (up day or down day)
    spike_signal = None
    vspike = vol_spike.tolist()
    cl = closes.tolist()
    hl_list = highs.tolist()
    lo_list = lows.tolist()
    for i in range(1, len(vspike)):
        if vspike[i] and _valid(cl[i]) and _valid(cl[i - 1]):
            spike_signal = "bullish" if cl[i] > cl[i - 1] else "bearish"

    # ── Support & Resistance ──────────────────────────────────────────────
    n_bars = len(cl)
    sw_win = 5  # bars each side for swing point detection

    swing_highs, swing_lows = [], []
    for i in range(sw_win, n_bars - sw_win):
        seg = cl[i - sw_win: i + sw_win + 1]
        if cl[i] == max(seg):
            swing_highs.append(i)
        if cl[i] == min(seg):
            swing_lows.append(i)

    # Resistance trendline: last 2 swing highs
    resistance_line = None
    if len(swing_highs) >= 2:
        rx1, rx2 = swing_highs[-2], swing_highs[-1]
        ry1, ry2 = cl[rx1], cl[rx2]
        rslope = (ry2 - ry1) / (rx2 - rx1) if rx2 != rx1 else 0
        resistance_line = {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2, "slope": rslope}

    # Support trendline: last 2 swing lows
    support_line = None
    if len(swing_lows) >= 2:
        sx1, sx2 = swing_lows[-2], swing_lows[-1]
        sy1, sy2 = cl[sx1], cl[sx2]
        sslope = (sy2 - sy1) / (sx2 - sx1) if sx2 != sx1 else 0
        support_line = {"x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2, "slope": sslope}

    # Trendline signal: price vs projected support at current bar
    trendline_signal = None
    if support_line:
        proj_y = support_line["y2"] + support_line["slope"] * (n_bars - 1 - support_line["x2"])
        trendline_signal = "above_support" if cl[-1] > proj_y else "below_support"

    # Fibonacci retracement (last 126 trading days of H/L)
    fib_hi = max(hl_list[-126:]) if len(hl_list) >= 126 else max(hl_list)
    fib_lo = min(lo_list[-126:]) if len(lo_list) >= 126 else min(lo_list)
    fib_rng = fib_hi - fib_lo
    fib_levels = {
        "0.0%":  round(fib_hi, 2),
        "23.6%": round(fib_hi - 0.236 * fib_rng, 2),
        "38.2%": round(fib_hi - 0.382 * fib_rng, 2),
        "50.0%": round(fib_hi - 0.500 * fib_rng, 2),
        "61.8%": round(fib_hi - 0.618 * fib_rng, 2),
        "78.6%": round(fib_hi - 0.786 * fib_rng, 2),
        "100%":  round(fib_lo, 2),
    }

    # Pivot Points (standard, from last completed bar)
    pp_h = float(highs.iloc[-1])
    pp_l = float(lows.iloc[-1])
    pp_c = float(closes.iloc[-1])
    pp = (pp_h + pp_l + pp_c) / 3
    pivot_points = {
        "PP": round(pp, 2),
        "R1": round(2 * pp - pp_l, 2),
        "R2": round(pp + (pp_h - pp_l), 2),
        "S1": round(2 * pp - pp_h, 2),
        "S2": round(pp - (pp_h - pp_l), 2),
    }

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
        "rsi": rsi.tolist(),
        "stoch_k": sk,
        "stoch_d": sd,
        "stoch_crossover": stoch_crossover,
        "volume": volume.tolist(),
        "vol_mean": vol_mean.tolist(),
        "vol_spike": vspike,
        "obv": obv_vals,
        "obv_trend": obv_trend,
        "spike_signal": spike_signal,
        "bb_upper": bb_upper.tolist(),
        "bb_lower": bb_lower.tolist(),
        "bb_mid": bb_mid.tolist(),
        "bb_signal": bb_signal,
        "atr": atr_vals,
        "atr_mean": atr_mean_vals,
        "atr_level": atr_level,
        "atr_ratio": atr_ratio,
        "support_line": support_line,
        "resistance_line": resistance_line,
        "trendline_signal": trendline_signal,
        "fib_levels": fib_levels,
        "pivot_points": pivot_points,
        "n_bars": n_bars,
    }


def run_prediction(ticker: str, timeframe: str = "1w", indicators: set | None = None) -> dict:
    if indicators is None:
        indicators = {"trend", "momentum", "volatility", "volume", "support", "fundamental"}

    base_price = get_current_price(ticker)

    try:
        tech = get_technical_indicators(ticker)
    except Exception:
        tech = None

    try:
        fund = get_fundamental_indicators(ticker) if "fundamental" in indicators else None
    except Exception:
        fund = None

    bullish_score = 0
    bearish_score = 0
    key_factors: list[str] = []

    if tech:
        last_sma50 = next((v for v in reversed(tech["sma50"]) if _valid(v)), None)
        last_sma200 = next((v for v in reversed(tech["sma200"]) if _valid(v)), None)
        last_macd = next((v for v in reversed(tech["macd_line"]) if _valid(v)), None)
        last_signal = next((v for v in reversed(tech["signal_line"]) if _valid(v)), None)

        if "trend" in indicators:
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

        if "momentum" in indicators:
            # RSI (14-period)
            last_rsi = next((v for v in reversed(tech["rsi"]) if _valid(v)), None)
            if last_rsi is not None:
                if last_rsi < 30:
                    bullish_score += 2
                    key_factors.append(f"RSI oversold ({last_rsi:.1f}) — potential reversal up")
                elif last_rsi > 70:
                    bearish_score += 2
                    key_factors.append(f"RSI overbought ({last_rsi:.1f}) — potential reversal down")
                elif last_rsi >= 50:
                    bullish_score += 1
                    key_factors.append(f"RSI bullish momentum ({last_rsi:.1f})")
                else:
                    bearish_score += 1
                    key_factors.append(f"RSI bearish momentum ({last_rsi:.1f})")

            # Stochastic Oscillator
            last_stoch_k = next((v for v in reversed(tech["stoch_k"]) if _valid(v)), None)
            if tech["stoch_crossover"] == "bullish":
                bullish_score += 1
                key_factors.append("Stochastic %K crossed above %D (bullish)")
            elif tech["stoch_crossover"] == "bearish":
                bearish_score += 1
                key_factors.append("Stochastic %K crossed below %D (bearish)")
            if last_stoch_k is not None:
                if last_stoch_k < 20:
                    bullish_score += 1
                    key_factors.append(f"Stochastic oversold (%K={last_stoch_k:.1f})")
                elif last_stoch_k > 80:
                    bearish_score += 1
                    key_factors.append(f"Stochastic overbought (%K={last_stoch_k:.1f})")

        if "volume" in indicators:
            # OBV trend
            if tech["obv_trend"] == "rising":
                bullish_score += 1
                key_factors.append("OBV rising — volume confirms buying pressure")
            elif tech["obv_trend"] == "falling":
                bearish_score += 1
                key_factors.append("OBV falling — volume confirms selling pressure")

            # Volume spike direction
            if tech["spike_signal"] == "bullish":
                bullish_score += 1
                key_factors.append("Volume spike on up day — strong buying interest")
            elif tech["spike_signal"] == "bearish":
                bearish_score += 1
                key_factors.append("Volume spike on down day — strong selling pressure")

        if "support" in indicators:
            # Trendlines
            if tech.get("trendline_signal") == "above_support":
                sl_data = tech.get("support_line", {})
                if sl_data and sl_data.get("slope", 0) > 0:
                    bullish_score += 1
                    key_factors.append("Price above rising support trendline")
                else:
                    key_factors.append("Price above support trendline")
            elif tech.get("trendline_signal") == "below_support":
                bearish_score += 1
                key_factors.append("Price broke below support trendline")

            # Pivot Points
            pp_val = tech["pivot_points"]["PP"]
            if base_price > pp_val:
                bullish_score += 1
                key_factors.append(f"Price above Pivot Point (${pp_val:.2f})")
            else:
                bearish_score += 1
                key_factors.append(f"Price below Pivot Point (${pp_val:.2f})")

        if "volatility" in indicators:
            # Bollinger Bands position
            bb_sig = tech.get("bb_signal")
            last_bb_upper = next((v for v in reversed(tech["bb_upper"]) if _valid(v)), None)
            last_bb_lower = next((v for v in reversed(tech["bb_lower"]) if _valid(v)), None)
            if bb_sig == "above_upper" and last_bb_upper:
                bearish_score += 1
                key_factors.append(f"Price above upper BB (${last_bb_upper:.2f}) — overbought")
            elif bb_sig == "below_lower" and last_bb_lower:
                bullish_score += 1
                key_factors.append(f"Price below lower BB (${last_bb_lower:.2f}) — oversold")

    if fund and "fundamental" in indicators:
        pe = fund.get("trailing_pe")
        if pe is not None:
            if pe < 15:
                bullish_score += 1
                key_factors.append(f"P/E attractive ({pe:.1f}x) — potentially undervalued")
            elif pe > 35:
                bearish_score += 1
                key_factors.append(f"P/E elevated ({pe:.1f}x) — potentially overvalued")

        rev_growth = fund.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth > 0.10:
                bullish_score += 1
                key_factors.append(f"Revenue growth strong ({rev_growth*100:+.1f}% YoY)")
            elif rev_growth < 0:
                bearish_score += 1
                key_factors.append(f"Revenue declining ({rev_growth*100:+.1f}% YoY)")

        earn_growth = fund.get("earnings_growth")
        if earn_growth is not None:
            if earn_growth > 0.15:
                bullish_score += 1
                key_factors.append(f"Earnings growth strong ({earn_growth*100:+.1f}% YoY)")
            elif earn_growth < 0:
                bearish_score += 1
                key_factors.append(f"Earnings declining ({earn_growth*100:+.1f}% YoY)")

        net_margin = fund.get("net_margin")
        if net_margin is not None and net_margin > 0.15:
            bullish_score += 1
            key_factors.append(f"Strong net margin ({net_margin*100:.1f}%)")

        roe = fund.get("roe")
        if roe is not None:
            if roe > 0.15:
                bullish_score += 1
                key_factors.append(f"Strong ROE ({roe*100:.1f}%)")
            elif roe < 0:
                bearish_score += 1
                key_factors.append(f"Negative ROE ({roe*100:.1f}%) — unprofitable")

        de = fund.get("debt_to_equity")
        if de is not None:
            de_ratio = de / 100
            if de_ratio < 0.5:
                bullish_score += 1
                key_factors.append(f"Low debt-to-equity ({de_ratio:.2f}x) — financially healthy")
            elif de_ratio > 2.0:
                bearish_score += 1
                key_factors.append(f"High debt-to-equity ({de_ratio:.2f}x) — heavily leveraged")

        current_ratio = fund.get("current_ratio")
        if current_ratio is not None:
            if current_ratio < 1.0:
                bearish_score += 1
                key_factors.append(f"Current ratio weak ({current_ratio:.2f}) — liquidity risk")

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

    # ATR-derived risk level and factor
    risk_level = tech["atr_level"] if tech else random.choice(["low", "medium", "high"])
    if tech:
        last_atr = next((v for v in reversed(tech["atr"]) if _valid(v)), None)
        if last_atr:
            ratio = tech["atr_ratio"]
            key_factors.append(
                f"ATR={last_atr:.2f} ({ratio:.1f}× avg) — "
                f"{'high' if ratio > 1.3 else 'low' if ratio < 0.8 else 'moderate'} volatility"
            )

    _ytd_days = (datetime.now() - datetime(datetime.now().year, 1, 1)).days or 1
    timeframe_days = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "ytd": _ytd_days, "1y": 365, "2y": 730, "5y": 1825}
    target_date = (datetime.now() + timedelta(days=timeframe_days.get(timeframe, 7))).strftime("%Y-%m-%d")

    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "direction": direction,
        "confidence": confidence,
        "current_price": base_price,
        "price_target": price_target,
        "target_date": target_date,
        "key_factors": key_factors[:6],
        "risk_level": risk_level,
        "technical": tech,
        "fundamental": fund,
        "indicators": sorted(indicators),
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
    fund = prediction.get("fundamental") or {}
    active = set(prediction.get("indicators", ["trend", "momentum", "volatility", "volume", "support", "fundamental"]))

    color_map = {"bullish": "#26a69a", "bearish": "#ef5350", "neutral": "#ffa726"}
    risk_colors = {"low": "#26a69a", "medium": "#ffa726", "high": "#ef5350"}
    main_color = color_map[direction]

    # Build rows dynamically from selected indicators (in display order)
    _PANEL_ORDER = [
        ("trend",       "full",  1.2),
        ("momentum",    "split", 1.2),
        ("volume",      "split", 1.2),
        ("support",     "full",  1.2),
        ("volatility",  "full",  2.0),
        ("fundamental", "full",  2.0),
    ]
    optional = [(cat, layout, h) for cat, layout, h in _PANEL_ORDER if cat in active]
    height_ratios = [2.5] + [h for _, _, h in optional] + [1.2]
    fig_height = round(sum(height_ratios) * (24 / 11.5), 1)

    fig = plt.figure(figsize=(14, fig_height), facecolor="#0d1117")
    change_pct = (target - current) / current * 100
    sign = "+" if change_pct >= 0 else ""
    fig.suptitle(
        f"{ticker} — Technical Analysis ({timeframe})  ·  Target ${target:,.2f} ({sign}{change_pct:.1f}%)",
        fontsize=15, fontweight="bold", color="white", y=0.99,
    )
    gs = gridspec.GridSpec(
        len(height_ratios), 2, figure=fig,
        height_ratios=height_ratios,
        hspace=0.55, wspace=0.35,
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

        if "trend" in active:
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

        if "volatility" in active:
            # Bollinger Bands overlay
            bbu = tech["bb_upper"][-display_n:]
            bbl = tech["bb_lower"][-display_n:]
            bb_idx = [i for i in range(len(bbu)) if _valid(bbu[i]) and _valid(bbl[i])]
            if bb_idx:
                bx  = [xs[i] for i in bb_idx]
                buy = [bbu[i] for i in bb_idx]
                bly = [bbl[i] for i in bb_idx]
                ax1.plot(bx, buy, color="#e879f9", linewidth=0.9,
                         linestyle=":", alpha=0.8, label="BB Upper")
                ax1.plot(bx, bly, color="#e879f9", linewidth=0.9,
                         linestyle=":", alpha=0.8, label="BB Lower")
                ax1.fill_between(bx, buy, bly, alpha=0.05, color="#e879f9")

        last_x = xs[-1]
        proj_len = max(4, display_n // 8)

        if "support" in active:
            # Trendline overlays (support = cyan dash-dot, resistance = pink dash-dot)
            for tl_key, tl_color, tl_label in [
                ("support_line",    "#22d3ee", "Support TL"),
                ("resistance_line", "#f472b6", "Resistance TL"),
            ]:
                tl = tech.get(tl_key)
                if tl:
                    n_all_tl = tech["n_bars"]
                    x2d = tl["x2"] - (n_all_tl - display_n)
                    if x2d >= 0:
                        x_ext = last_x + proj_len
                        x1d = max(tl["x1"] - (n_all_tl - display_n), 0)
                        tl_xs = [x1d, x_ext]
                        tl_ys = [
                            tl["y2"] + tl["slope"] * (x1d - x2d),
                            tl["y2"] + tl["slope"] * (x_ext - x2d),
                        ]
                        ax1.plot(tl_xs, tl_ys, color=tl_color, linewidth=1.1,
                                 linestyle="-.", alpha=0.85, label=tl_label)

        if "trend" in active:
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

    row = 1
    for _cat, _, _ in optional:

        if _cat == "trend":
            # ── MACD (12, 26, 9) ──────────────────────────────────────────
            ax2 = fig.add_subplot(gs[row, :])
            ax2.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                display_n = min(126, n_all)
                xs_m = list(range(display_n))
                ml = tech["macd_line"][-display_n:]
                sl = tech["signal_line"][-display_n:]
                hl = tech["histogram"][-display_n:]
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
                    _trend = "▲ Bullish" if last_m > last_s else "▼ Bearish"
                    status = f"{_trend}   MACD {last_m:+.3f}   Signal {last_s:+.3f}"
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

        elif _cat == "momentum":
            # ── RSI (14) ──────────────────────────────────────────────────
            ax3 = fig.add_subplot(gs[row, 0])
            ax3.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                display_n = min(126, n_all)
                xs_r = list(range(display_n))
                rsi_vals = tech["rsi"][-display_n:]
                valid_rsi = [(xs_r[i], rsi_vals[i]) for i in range(len(rsi_vals)) if _valid(rsi_vals[i])]
                if valid_rsi:
                    rx = [p[0] for p in valid_rsi]
                    ry = [p[1] for p in valid_rsi]
                    ax3.plot(rx, ry, color="#38bdf8", linewidth=1.4, label="RSI (14)")
                    ax3.fill_between(rx, ry, 70, where=[v > 70 for v in ry],
                                     color="#ef5350", alpha=0.25, label="Overbought")
                    ax3.fill_between(rx, ry, 30, where=[v < 30 for v in ry],
                                     color="#26a69a", alpha=0.25, label="Oversold")
                ax3.axhline(70, color="#ef5350", linewidth=0.8, linestyle="--", alpha=0.7)
                ax3.axhline(50, color="#888888", linewidth=0.6, linestyle=":")
                ax3.axhline(30, color="#26a69a", linewidth=0.8, linestyle="--", alpha=0.7)
                ax3.set_ylim(0, 100)
                ax3.set_yticks([30, 50, 70])
                ax3.set_yticklabels(["30", "50", "70"], color="gray", fontsize=7)
                last_rsi = next((v for v in reversed(rsi_vals) if _valid(v)), None)
                rsi_status = f"  ·  {last_rsi:.1f}" if last_rsi is not None else ""
                rsi_label = (" — Overbought" if last_rsi and last_rsi > 70
                             else " — Oversold" if last_rsi and last_rsi < 30 else "")
                ax3.set_title(f"RSI (14){rsi_status}{rsi_label}", color="white", fontsize=10, pad=6)
                ax3.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                           edgecolor="#30363d", labelcolor="white", framealpha=0.8)
            else:
                ax3.text(0.5, 0.5, "RSI unavailable", ha="center", va="center",
                         color="gray", transform=ax3.transAxes)
                ax3.set_title("RSI (14)", color="white", fontsize=10, pad=6)
            ax3.tick_params(colors="gray", labelsize=7)
            for spine in ax3.spines.values():
                spine.set_edgecolor("#30363d")

            # ── Stochastic (14, 3) ─────────────────────────────────────────
            ax4 = fig.add_subplot(gs[row, 1])
            ax4.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                display_n = min(126, n_all)
                xs_s = list(range(display_n))
                sk_vals = tech["stoch_k"][-display_n:]
                sd_vals = tech["stoch_d"][-display_n:]
                valid_sk = [(xs_s[i], sk_vals[i]) for i in range(len(sk_vals)) if _valid(sk_vals[i])]
                valid_sd = [(xs_s[i], sd_vals[i]) for i in range(len(sd_vals)) if _valid(sd_vals[i])]
                if valid_sk:
                    skx = [p[0] for p in valid_sk]
                    sky = [p[1] for p in valid_sk]
                    ax4.plot(skx, sky, color="#a78bfa", linewidth=1.4, label="%K (14)")
                    ax4.fill_between(skx, sky, 80, where=[v > 80 for v in sky],
                                     color="#ef5350", alpha=0.20)
                    ax4.fill_between(skx, sky, 20, where=[v < 20 for v in sky],
                                     color="#26a69a", alpha=0.20)
                if valid_sd:
                    ax4.plot([p[0] for p in valid_sd], [p[1] for p in valid_sd],
                             color="#ffa726", linewidth=1.1, linestyle="--", label="%D (3)")
                ax4.axhline(80, color="#ef5350", linewidth=0.8, linestyle="--", alpha=0.7)
                ax4.axhline(50, color="#888888", linewidth=0.6, linestyle=":")
                ax4.axhline(20, color="#26a69a", linewidth=0.8, linestyle="--", alpha=0.7)
                ax4.set_ylim(0, 100)
                ax4.set_yticks([20, 50, 80])
                ax4.set_yticklabels(["20", "50", "80"], color="gray", fontsize=7)
                last_sk = next((v for v in reversed(sk_vals) if _valid(v)), None)
                last_sd = next((v for v in reversed(sd_vals) if _valid(v)), None)
                stoch_status = (f"  ·  %K={last_sk:.1f}  %D={last_sd:.1f}"
                                if last_sk is not None and last_sd is not None else "")
                stoch_zone = (" — Overbought" if last_sk and last_sk > 80
                              else " — Oversold" if last_sk and last_sk < 20 else "")
                ax4.set_title(f"Stochastic (14, 3){stoch_status}{stoch_zone}",
                              color="white", fontsize=10, pad=6)
                ax4.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                           edgecolor="#30363d", labelcolor="white", framealpha=0.8)
            else:
                ax4.text(0.5, 0.5, "Stochastic unavailable", ha="center", va="center",
                         color="gray", transform=ax4.transAxes)
                ax4.set_title("Stochastic (14, 3)", color="white", fontsize=10, pad=6)
            ax4.tick_params(colors="gray", labelsize=7)
            for spine in ax4.spines.values():
                spine.set_edgecolor("#30363d")

        elif _cat == "volume":
            # ── Volume + Spike markers ─────────────────────────────────────
            ax5 = fig.add_subplot(gs[row, 0])
            ax5.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                display_n = min(126, n_all)
                xs_v = list(range(display_n))
                vol_vals  = tech["volume"][-display_n:]
                vm_vals   = tech["vol_mean"][-display_n:]
                spk_vals  = tech["vol_spike"][-display_n:]
                cl_vals   = tech["closes"][-display_n:]
                prev_cl   = tech["closes"][-(display_n + 1):-1]
                bar_colors = []
                for i in range(len(vol_vals)):
                    if i == 0 or not _valid(cl_vals[i]) or not _valid(prev_cl[i]):
                        bar_colors.append("#888888")
                    elif cl_vals[i] >= prev_cl[i]:
                        bar_colors.append("#26a69a")
                    else:
                        bar_colors.append("#ef5350")
                ax5.bar(xs_v, vol_vals, color=bar_colors, alpha=0.7, width=1.0)
                valid_vm = [(xs_v[i], vm_vals[i]) for i in range(len(vm_vals)) if _valid(vm_vals[i])]
                if valid_vm:
                    ax5.plot([p[0] for p in valid_vm], [p[1] for p in valid_vm],
                             color="#ffa726", linewidth=1.1, label="Vol MA(20)")
                spike_xs = [xs_v[i] for i in range(len(spk_vals)) if spk_vals[i]]
                spike_ys = [vol_vals[i] for i in spike_xs]
                if spike_xs:
                    ax5.scatter(spike_xs, spike_ys, color="#ffd700", s=40,
                                zorder=5, marker="^", label="Spike")
                ax5.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
                )
                ax5.set_title("Volume  (green=up day, red=down day, ▲=spike)",
                              color="white", fontsize=10, pad=6)
                ax5.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                           edgecolor="#30363d", labelcolor="white", framealpha=0.8)
            else:
                ax5.text(0.5, 0.5, "Volume unavailable", ha="center", va="center",
                         color="gray", transform=ax5.transAxes)
                ax5.set_title("Volume", color="white", fontsize=10, pad=6)
            ax5.tick_params(colors="gray", labelsize=7)
            for spine in ax5.spines.values():
                spine.set_edgecolor("#30363d")

            # ── OBV ────────────────────────────────────────────────────────
            ax6 = fig.add_subplot(gs[row, 1])
            ax6.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                display_n = min(126, n_all)
                xs_o = list(range(display_n))
                obv_vals = tech["obv"][-display_n:]
                valid_obv = [(xs_o[i], obv_vals[i]) for i in range(len(obv_vals)) if _valid(obv_vals[i])]
                if valid_obv:
                    ox = [p[0] for p in valid_obv]
                    oy = [p[1] for p in valid_obv]
                    obv_color = "#26a69a" if tech["obv_trend"] == "rising" else "#ef5350"
                    ax6.plot(ox, oy, color=obv_color, linewidth=1.4, label="OBV")
                    ax6.fill_between(ox, oy, min(oy), alpha=0.12, color=obv_color)
                trend_label = tech.get("obv_trend", "").capitalize() or "N/A"
                ax6.set_title(f"OBV — {trend_label}", color="white", fontsize=10, pad=6)
                ax6.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x/1e3:.0f}K")
                )
                ax6.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                           edgecolor="#30363d", labelcolor="white", framealpha=0.8)
            else:
                ax6.text(0.5, 0.5, "OBV unavailable", ha="center", va="center",
                         color="gray", transform=ax6.transAxes)
                ax6.set_title("OBV", color="white", fontsize=10, pad=6)
            ax6.tick_params(colors="gray", labelsize=7)
            for spine in ax6.spines.values():
                spine.set_edgecolor("#30363d")

        elif _cat == "support":
            # ── Support & Resistance ────────────────────────────────────────
            ax7 = fig.add_subplot(gs[row, :])
            ax7.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                sr_n = min(60, n_all)
                sr_xs = list(range(sr_n))
                sr_cl = tech["closes"][-sr_n:]
                sr_offset = n_all - sr_n
                ax7.plot(sr_xs, sr_cl, color=main_color, linewidth=1.4, label="Price", zorder=3)
                ax7.fill_between(sr_xs, sr_cl, alpha=0.07, color=main_color)
                fib_colors = {
                    "0.0%":  "#94a3b8", "23.6%": "#60a5fa", "38.2%": "#818cf8",
                    "50.0%": "#a78bfa", "61.8%": "#c084fc", "78.6%": "#e879f9", "100%": "#94a3b8",
                }
                key_fibs = {"38.2%", "50.0%", "61.8%"}
                for label, price_val in tech["fib_levels"].items():
                    lw = 1.2 if label in key_fibs else 0.7
                    ls = "--" if label in key_fibs else ":"
                    ax7.axhline(price_val, color=fib_colors.get(label, "#888"),
                                linewidth=lw, linestyle=ls, alpha=0.8)
                    ax7.text(sr_n - 0.5, price_val, f" Fib {label} ${price_val:.2f}",
                             color=fib_colors.get(label, "#888"), fontsize=6, va="center")
                pv = tech["pivot_points"]
                pv_styles = {
                    "R2": ("#ef5350", "-",  0.8), "R1": ("#ef9a9a", "--", 0.8),
                    "PP": ("#ffd700", "-",  1.0),
                    "S1": ("#a5d6a7", "--", 0.8), "S2": ("#26a69a", "-",  0.8),
                }
                for name, price_val in pv.items():
                    col, ls, lw = pv_styles[name]
                    ax7.axhline(price_val, color=col, linewidth=lw, linestyle=ls, alpha=0.85)
                    ax7.text(0.5, price_val, f" {name} ${price_val:.2f}",
                             color=col, fontsize=6, va="center")
                for tl_key, tl_color, tl_lbl in [
                    ("support_line", "#22d3ee", "Support"), ("resistance_line", "#f472b6", "Resist."),
                ]:
                    tl = tech.get(tl_key)
                    if tl:
                        x2d = tl["x2"] - sr_offset
                        if x2d >= 0:
                            x1d = max(tl["x1"] - sr_offset, 0)
                            x_end = sr_n - 1
                            tl_ys = [
                                tl["y2"] + tl["slope"] * (x1d - x2d),
                                tl["y2"] + tl["slope"] * (x_end - x2d),
                            ]
                            ax7.plot([x1d, x_end], tl_ys, color=tl_color,
                                     linewidth=1.2, linestyle="-.", alpha=0.9, label=tl_lbl)
                ax7.set_title("Support & Resistance  (Fib Retracement · Pivot Points · Trendlines)",
                              color="white", fontsize=10, pad=6)
                ax7.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                           edgecolor="#30363d", labelcolor="white", framealpha=0.8)
            else:
                ax7.text(0.5, 0.5, "S&R unavailable", ha="center", va="center",
                         color="gray", transform=ax7.transAxes)
                ax7.set_title("Support & Resistance", color="white", fontsize=10, pad=6)
            ax7.tick_params(colors="gray", labelsize=7)
            for spine in ax7.spines.values():
                spine.set_edgecolor("#30363d")

        elif _cat == "volatility":
            # ── ATR (14) ────────────────────────────────────────────────────
            ax8_atr = fig.add_subplot(gs[row, :])
            ax8_atr.set_facecolor("#161b22")
            if tech:
                n_all = len(tech["xs"])
                display_n = min(126, n_all)
                xs_a = list(range(display_n))
                atr_vals  = tech["atr"][-display_n:]
                atrm_vals = tech["atr_mean"][-display_n:]
                valid_atr  = [(xs_a[i], atr_vals[i])  for i in range(len(atr_vals))  if _valid(atr_vals[i])]
                valid_atrm = [(xs_a[i], atrm_vals[i]) for i in range(len(atrm_vals)) if _valid(atrm_vals[i])]
                if valid_atr:
                    ax_x = [p[0] for p in valid_atr]
                    ax_y = [p[1] for p in valid_atr]
                    atr_color = (
                        "#ef5350" if tech["atr_level"] == "high"
                        else "#26a69a" if tech["atr_level"] == "low"
                        else "#ffa726"
                    )
                    ax8_atr.plot(ax_x, ax_y, color=atr_color, linewidth=1.4, label="ATR (14)")
                if valid_atrm:
                    ax8_atr.plot([p[0] for p in valid_atrm], [p[1] for p in valid_atrm],
                                 color="#888888", linewidth=1.0, linestyle="--", alpha=0.7, label="ATR MA(20)")
                if valid_atr and valid_atrm:
                    atr_dict  = {p[0]: p[1] for p in valid_atr}
                    atrm_dict = {p[0]: p[1] for p in valid_atrm}
                    common = [x for x in atrm_dict if x in atr_dict]
                    if common:
                        ax8_atr.fill_between(
                            common,
                            [atr_dict[x] for x in common],
                            [atrm_dict[x] for x in common],
                            where=[atr_dict[x] >= atrm_dict[x] for x in common],
                            alpha=0.18, color="#ef5350", label="High vol",
                        )
                        ax8_atr.fill_between(
                            common,
                            [atr_dict[x] for x in common],
                            [atrm_dict[x] for x in common],
                            where=[atr_dict[x] < atrm_dict[x] for x in common],
                            alpha=0.18, color="#26a69a", label="Low vol",
                        )
                last_atr_v = next((v for v in reversed(atr_vals) if _valid(v)), None)
                ratio = tech.get("atr_ratio", 1.0)
                level = tech.get("atr_level", "medium").capitalize()
                atr_status = f"  ·  {last_atr_v:.2f}  ({ratio:.1f}× avg)  —  {level} volatility" if last_atr_v else ""
                ax8_atr.set_title(f"ATR (14){atr_status}", color="white", fontsize=10, pad=6)
                ax8_atr.legend(fontsize=6, loc="upper left", facecolor="#161b22",
                               edgecolor="#30363d", labelcolor="white", framealpha=0.8, ncol=4)
            else:
                ax8_atr.text(0.5, 0.5, "ATR unavailable", ha="center", va="center",
                             color="gray", transform=ax8_atr.transAxes)
                ax8_atr.set_title("ATR (14)", color="white", fontsize=10, pad=6)
            ax8_atr.tick_params(colors="gray", labelsize=7)
            for spine in ax8_atr.spines.values():
                spine.set_edgecolor("#30363d")

        elif _cat == "fundamental":
            # ── Fundamental Indicators ──────────────────────────────────────
            ax_f = fig.add_subplot(gs[row, :])
            ax_f.set_facecolor("#161b22")
            ax_f.axis("off")
            ax_f.set_xlim(0, 1)
            ax_f.set_ylim(0, 1)

            sig_colors = {"bull": "#26a69a", "bear": "#ef5350", "neutral": "#ffa726", "none": "#444444"}

            def _fsig(val, bull_thresh, bear_thresh, higher_is_bull=True):
                if val is None:
                    return "none"
                if higher_is_bull:
                    return "bull" if val >= bull_thresh else ("bear" if val <= bear_thresh else "neutral")
                return "bull" if val <= bull_thresh else ("bear" if val >= bear_thresh else "neutral")

            de_ratio = (fund.get("debt_to_equity") or 0) / 100

            metrics = [
                ("P/E (TTM)",    fund.get("trailing_pe"),      lambda v: _fsig(v, 99, 15, False),    lambda v: f"{v:.1f}×"),
                ("Fwd P/E",      fund.get("forward_pe"),       lambda v: _fsig(v, 99, 15, False),    lambda v: f"{v:.1f}×"),
                ("P/B",          fund.get("price_to_book"),    lambda v: _fsig(v, 99, 2, False),     lambda v: f"{v:.1f}×"),
                ("P/S",          fund.get("price_to_sales"),   lambda v: _fsig(v, 99, 3, False),     lambda v: f"{v:.1f}×"),
                ("EV/EBITDA",    fund.get("ev_ebitda"),        lambda v: _fsig(v, 99, 10, False),    lambda v: f"{v:.1f}×"),
                ("PEG",          fund.get("peg_ratio"),        lambda v: _fsig(v, 99, 1, False),     lambda v: f"{v:.2f}"),
                ("Rev Growth",   fund.get("revenue_growth"),   lambda v: _fsig(v, 0.10, -0.01),      lambda v: f"{v*100:+.1f}%"),
                ("EPS Growth",   fund.get("earnings_growth"),  lambda v: _fsig(v, 0.15, -0.01),      lambda v: f"{v*100:+.1f}%"),
                ("Net Margin",   fund.get("net_margin"),       lambda v: _fsig(v, 0.15, 0.05),       lambda v: f"{v*100:.1f}%"),
                ("Op Margin",    fund.get("operating_margin"), lambda v: _fsig(v, 0.15, 0.05),       lambda v: f"{v*100:.1f}%"),
                ("ROE",          fund.get("roe"),              lambda v: _fsig(v, 0.15, 0),           lambda v: f"{v*100:.1f}%"),
                ("D/E",          de_ratio if fund.get("debt_to_equity") else None,
                                                               lambda v: _fsig(v, 99, 0.5, False),   lambda v: f"{v:.2f}×"),
                ("Curr Ratio",   fund.get("current_ratio"),   lambda v: _fsig(v, 2.0, 1.0),          lambda v: f"{v:.2f}"),
                ("Div Yield",    fund.get("dividend_yield"),  lambda v: _fsig(v, 0.03, 0),           lambda v: f"{v*100:.1f}%"),
                ("Short Ratio",  fund.get("short_ratio"),     lambda v: _fsig(v, 99, 2.0, False),    lambda v: f"{v:.1f}d"),
            ]

            n_cols = 5
            col_w = 1.0 / n_cols
            box_h = 0.38
            row_gap = 0.46
            top = 0.92

            for idx, (label, val, sig_fn, fmt_fn) in enumerate(metrics):
                c = idx % n_cols
                r = idx // n_cols
                x0 = c * col_w + 0.005
                y0 = top - r * row_gap

                sig = sig_fn(val) if val is not None else "none"
                color = sig_colors[sig]
                display = fmt_fn(val) if val is not None else "N/A"

                ax_f.add_patch(mpatches.FancyBboxPatch(
                    (x0, y0 - box_h), col_w - 0.01, box_h,
                    boxstyle="round,pad=0.01", facecolor=color, edgecolor="none",
                    alpha=0.15, transform=ax_f.transAxes,
                ))
                ax_f.text(x0 + (col_w - 0.01) / 2, y0 - 0.10, label,
                          ha="center", va="center", color="#aaaaaa", fontsize=7,
                          transform=ax_f.transAxes)
                ax_f.text(x0 + (col_w - 0.01) / 2, y0 - 0.27, display,
                          ha="center", va="center",
                          color=color if val is not None else "#555555",
                          fontsize=9, fontweight="bold", transform=ax_f.transAxes)

            ax_f.set_title("Fundamental Indicators", color="white", fontsize=10, pad=6)

        row += 1

    # ── Confidence + Risk gauge ────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[row, 0])
    ax8.set_facecolor("#161b22")
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis("off")

    theta = np.linspace(np.pi, np.pi * (1 - confidence), 100)
    ax8.plot(
        0.3 + 0.22 * np.cos(np.linspace(np.pi, 0, 100)),
        0.55 + 0.22 * np.sin(np.linspace(np.pi, 0, 100)),
        color="#30363d", linewidth=10, solid_capstyle="round",
    )
    ax8.plot(
        0.3 + 0.22 * np.cos(theta),
        0.55 + 0.22 * np.sin(theta),
        color=main_color, linewidth=10, solid_capstyle="round",
    )
    ax8.text(0.3, 0.50, f"{int(confidence * 100)}%", ha="center", va="center",
             color="white", fontsize=18, fontweight="bold")
    ax8.text(0.3, 0.20, "Confidence", ha="center", color="gray", fontsize=9)

    risk_color = risk_colors[risk]
    pill = mpatches.FancyBboxPatch(
        (0.60, 0.42), 0.30, 0.18,
        boxstyle="round,pad=0.02", facecolor=risk_color, edgecolor="none", alpha=0.25,
    )
    ax8.add_patch(pill)
    ax8.text(0.75, 0.51, f"Risk: {risk.upper()}", ha="center", va="center",
             color=risk_color, fontsize=10, fontweight="bold")
    direction_icon = {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "◆ NEUTRAL"}[direction]
    ax8.text(0.75, 0.28, direction_icon, ha="center", color=main_color,
             fontsize=11, fontweight="bold")
    ax8.set_title("Confidence & Risk", color="white", fontsize=10, pad=6)

    # ── Panel 10: Technical signal factors ────────────────────────────────
    ax9 = fig.add_subplot(gs[row, 1])
    ax9.set_facecolor("#161b22")
    if factors:
        y_pos = range(len(factors))
        weights = [random.uniform(0.6, 1.0) for _ in factors]
        ax9.barh(list(y_pos), weights, color=main_color, alpha=0.8,
                 edgecolor="#30363d", height=0.5)
        ax9.set_yticks(list(y_pos))
        ax9.set_yticklabels(
            [f[:30] + "…" if len(f) > 30 else f for f in factors],
            color="white", fontsize=7,
        )
        ax9.set_xlim(0, 1.2)
        ax9.set_xticks([0, 0.5, 1.0])
        ax9.set_xticklabels(["Low", "Med", "High"], color="gray", fontsize=7)
    else:
        ax9.text(0.5, 0.5, "No signals", ha="center", va="center",
                 color="gray", transform=ax9.transAxes)
    ax9.set_title("Technical Signal Factors", color="white", fontsize=10, pad=6)
    ax9.tick_params(colors="gray")
    for spine in ax9.spines.values():
        spine.set_edgecolor("#30363d")

    chart_path = os.path.join(charts_dir, f"{ticker}_{timeframe}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Chart saved: {chart_path}")
    return chart_path


def predict_stock(ticker: str, timeframe: str = "1w", md_file=None, charts_dir: str = "charts", model: str = "claude-sonnet-4-6", indicators: set | None = None) -> None:
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
                prediction = run_prediction(ticker_input, tf_input, indicators=indicators)
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
    _all_indicators = ["trend", "momentum", "volatility", "volume", "support", "fundamental"]

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
            "  python stock_predictor.py --tickers AAPL --indicators trend momentum\n"
            "  python stock_predictor.py --tickers TSLA --indicators volatility volume\n"
            "  python stock_predictor.py --tickers MSFT --indicators fundamental\n"
        ),
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        default=["AAPL", "TSLA", "INTC"],
        help="one or more stock ticker symbols (default: AAPL TSLA INTC)",
    )
    parser.add_argument(
        "--timeframe", choices=["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "2y", "5y"], default=None,
        help="prediction timeframe for all tickers (default: 1w)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        metavar="MODEL",
        help="Claude model ID to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--indicators", nargs="+",
        choices=_all_indicators,
        default=None,
        metavar="INDICATOR",
        help=(
            "indicator categories to include: trend momentum volatility volume support fundamental "
            "(default: all). Example: --indicators trend momentum fundamental"
        ),
    )
    args = parser.parse_args()
    active_indicators = set(args.indicators) if args.indicators else set(_all_indicators)

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
        ind_str = ", ".join(sorted(active_indicators))
        f.write(f"**Tickers:** {tickers_str}  \n**Timeframe:** {tf_str}  \n**Model:** {args.model}  \n**Indicators:** {ind_str}\n\n")
        f.write("---\n\n")

        default_timeframes = {"AAPL": "1w", "TSLA": "1m", "INTC": "1m"}
        for ticker in args.tickers:
            tf = args.timeframe or default_timeframes.get(ticker.upper(), "1w")
            predict_stock(ticker, tf, md_file=f, charts_dir=charts_dir, model=args.model, indicators=active_indicators)

    print(f"\nResults saved to: {run_dir}/")
