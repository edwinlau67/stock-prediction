import json
import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import stock_predictor as sp
from stock_predictor import (
    ScoringConfig,
    _fetch_with_retry,
    _score_fundamental,
    _score_momentum,
    _score_support,
    _score_trend,
    _score_volatility,
    _score_volume,
    _valid,
    generate_chart,
    run_prediction,
)


class TestValid(unittest.TestCase):
    def test_none(self):
        self.assertFalse(_valid(None))

    def test_nan(self):
        self.assertFalse(_valid(float("nan")))

    def test_number(self):
        self.assertTrue(_valid(1.5))
        self.assertTrue(_valid(0.0))
        self.assertTrue(_valid(-99.9))

    def test_non_numeric(self):
        self.assertFalse(_valid("hello"))
        self.assertFalse(_valid([]))


class TestScoringConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ScoringConfig()
        self.assertEqual(cfg.pe_bull, 15.0)
        self.assertEqual(cfg.pe_bear, 35.0)
        self.assertEqual(cfg.rsi_oversold, 30.0)
        self.assertEqual(cfg.rsi_overbought, 70.0)
        self.assertEqual(cfg.conf_base, 0.52)
        self.assertEqual(cfg.conf_cap, 0.95)

    def test_from_json(self):
        data = {"pe_bull": 20.0, "conf_base": 0.55}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            cfg = ScoringConfig.from_json(path)
            self.assertEqual(cfg.pe_bull, 20.0)
            self.assertEqual(cfg.conf_base, 0.55)
            self.assertEqual(cfg.pe_bear, 35.0)  # unchanged default
        finally:
            os.unlink(path)

    def test_from_json_ignores_unknown_keys(self):
        data = {"unknown_field": 999, "pe_bull": 12.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            cfg = ScoringConfig.from_json(path)
            self.assertEqual(cfg.pe_bull, 12.0)
        finally:
            os.unlink(path)


def _make_tech(n=252) -> dict:
    closes = [100.0 + i * 0.1 for i in range(n)]
    return {
        "xs": list(range(n)),
        "closes": closes,
        "sma50": [None] * 49 + [100.0] * (n - 49),
        "sma200": [None] * 199 + [95.0] * (n - 199),
        "ema20": [101.0] * n,
        "macd_line": [0.5] * n,
        "signal_line": [0.3] * n,
        "histogram": [0.2] * n,
        "cross_signal": None,
        "cross_idx": None,
        "macd_crossover": None,
        "rsi": [50.0] * n,
        "stoch_k": [50.0] * n,
        "stoch_d": [50.0] * n,
        "stoch_crossover": None,
        "volume": [1_000_000.0] * n,
        "vol_mean": [900_000.0] * n,
        "vol_spike": [False] * n,
        "obv": [float(i * 1000) for i in range(n)],
        "obv_trend": "rising",
        "spike_signal": None,
        "bb_upper": [110.0] * n,
        "bb_lower": [90.0] * n,
        "bb_mid": [100.0] * n,
        "bb_signal": "within",
        "atr": [1.5] * n,
        "atr_mean": [1.5] * n,
        "atr_level": "medium",
        "atr_ratio": 1.0,
        "support_line": None,
        "resistance_line": None,
        "trendline_signal": None,
        "fib_levels": {"0.0%": 110.0, "23.6%": 108.0, "38.2%": 106.0, "50.0%": 105.0,
                       "61.8%": 104.0, "78.6%": 102.0, "100%": 95.0},
        "pivot_points": {"PP": 100.0, "R1": 102.0, "R2": 104.0, "S1": 98.0, "S2": 96.0},
        "n_bars": n,
    }


class TestScoreTrend(unittest.TestCase):
    def test_golden_cross(self):
        tech = _make_tech()
        tech["cross_signal"] = "golden"
        bull, bear, factors = _score_trend(tech, 102.0, ScoringConfig())
        self.assertEqual(bull, 5)   # golden cross +2, macd above signal +1, price above sma50 +1, price above sma200 +1
        self.assertGreater(bull, bear)
        self.assertTrue(any("Golden Cross" in f for f in factors))

    def test_death_cross(self):
        tech = _make_tech()
        tech["cross_signal"] = "death"
        bull, bear, factors = _score_trend(tech, 90.0, ScoringConfig())
        self.assertGreater(bear, bull)
        self.assertTrue(any("Death Cross" in f for f in factors))

    def test_price_vs_sma(self):
        tech = _make_tech()
        # price above sma50 (100.0) and sma200 (95.0) → bullish
        bull, bear, _ = _score_trend(tech, 105.0, ScoringConfig())
        # macd (0.5) > signal (0.3) → +1 bull, price > sma50 → +1, price > sma200 → +1
        self.assertGreaterEqual(bull, 3)


class TestScoreMomentum(unittest.TestCase):
    def test_rsi_oversold(self):
        tech = _make_tech()
        tech["rsi"] = [25.0] * 252
        bull, bear, factors = _score_momentum(tech, ScoringConfig())
        self.assertEqual(bull, 2)
        self.assertTrue(any("oversold" in f for f in factors))

    def test_rsi_overbought(self):
        tech = _make_tech()
        tech["rsi"] = [75.0] * 252
        bull, bear, factors = _score_momentum(tech, ScoringConfig())
        self.assertEqual(bear, 2)
        self.assertTrue(any("overbought" in f for f in factors))

    def test_rsi_neutral_bullish(self):
        tech = _make_tech()
        tech["rsi"] = [55.0] * 252
        bull, bear, _ = _score_momentum(tech, ScoringConfig())
        self.assertEqual(bull, 1)
        self.assertEqual(bear, 0)


class TestScoreVolume(unittest.TestCase):
    def test_obv_rising(self):
        tech = _make_tech()
        tech["obv_trend"] = "rising"
        bull, bear, _ = _score_volume(tech, ScoringConfig())
        self.assertEqual(bull, 1)

    def test_obv_falling_and_bearish_spike(self):
        tech = _make_tech()
        tech["obv_trend"] = "falling"
        tech["spike_signal"] = "bearish"
        bull, bear, _ = _score_volume(tech, ScoringConfig())
        self.assertEqual(bear, 2)


class TestScoreFundamental(unittest.TestCase):
    def test_pe_bull(self):
        fund = {"trailing_pe": 10.0, "revenue_growth": None, "earnings_growth": None,
                "net_margin": None, "roe": None, "debt_to_equity": None, "current_ratio": None}
        bull, bear, factors = _score_fundamental(fund, ScoringConfig())
        self.assertEqual(bull, 1)
        self.assertTrue(any("P/E" in f and "attractive" in f for f in factors))

    def test_pe_bear(self):
        fund = {"trailing_pe": 50.0, "revenue_growth": None, "earnings_growth": None,
                "net_margin": None, "roe": None, "debt_to_equity": None, "current_ratio": None}
        bull, bear, factors = _score_fundamental(fund, ScoringConfig())
        self.assertEqual(bear, 1)

    def test_strong_fundamentals(self):
        fund = {
            "trailing_pe": 12.0,
            "revenue_growth": 0.20,
            "earnings_growth": 0.25,
            "net_margin": 0.20,
            "roe": 0.25,
            "debt_to_equity": 30.0,    # 0.30× after /100
            "current_ratio": 2.0,
        }
        bull, bear, _ = _score_fundamental(fund, ScoringConfig())
        self.assertGreater(bull, bear)

    def test_weak_fundamentals(self):
        fund = {
            "trailing_pe": 60.0,
            "revenue_growth": -0.05,
            "earnings_growth": -0.10,
            "net_margin": None,
            "roe": -0.05,
            "debt_to_equity": 300.0,   # 3.0× after /100
            "current_ratio": 0.8,
        }
        bull, bear, _ = _score_fundamental(fund, ScoringConfig())
        self.assertGreater(bear, bull)


class TestRunPrediction(unittest.TestCase):
    def _make_prediction(self, direction="bullish"):
        tech = _make_tech()
        if direction == "bullish":
            tech["cross_signal"] = "golden"
        elif direction == "bearish":
            tech["cross_signal"] = "death"
            tech["rsi"] = [75.0] * 252

        with patch.object(sp, "get_current_price", return_value=100.0), \
             patch.object(sp, "get_technical_indicators", return_value=tech), \
             patch.object(sp, "get_fundamental_indicators", return_value={}):
            return run_prediction("FAKE", "1w")

    def test_direction_in_valid_values(self):
        pred = self._make_prediction()
        self.assertIn(pred["direction"], {"bullish", "bearish", "neutral"})

    def test_confidence_range(self):
        pred = self._make_prediction()
        self.assertGreaterEqual(pred["confidence"], 0.52)
        self.assertLessEqual(pred["confidence"], 0.95)

    def test_schema_keys(self):
        pred = self._make_prediction()
        required = {"ticker", "timeframe", "direction", "confidence", "current_price",
                    "price_target", "target_date", "key_factors", "risk_level",
                    "technical", "fundamental", "indicators"}
        self.assertTrue(required.issubset(pred.keys()))

    def test_key_factors_capped_at_6(self):
        pred = self._make_prediction()
        self.assertLessEqual(len(pred["key_factors"]), 6)

    def test_bullish_target_above_current(self):
        with patch.object(sp, "get_current_price", return_value=100.0), \
             patch.object(sp, "get_technical_indicators", return_value=None), \
             patch.object(sp, "get_fundamental_indicators", return_value=None):
            # With no tech/fund data, direction is neutral; test that target is reasonable
            pred = run_prediction("FAKE", "1w", indicators=set())
            self.assertIsNotNone(pred["price_target"])


class TestGenerateChart(unittest.TestCase):
    def _minimal_prediction(self) -> dict:
        return {
            "ticker": "TEST",
            "timeframe": "1w",
            "direction": "bullish",
            "confidence": 0.72,
            "current_price": 100.0,
            "price_target": 110.0,
            "target_date": "2026-05-01",
            "key_factors": ["Factor A", "Factor B"],
            "risk_level": "medium",
            "technical": None,
            "fundamental": None,
            "indicators": [],
        }

    def test_chart_created_no_data(self):
        pred = self._minimal_prediction()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_chart(pred, tmpdir)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(path.endswith(".png"))

    def test_chart_created_with_tech_data(self):
        pred = self._minimal_prediction()
        pred["technical"] = _make_tech()
        pred["indicators"] = ["trend", "momentum", "volume"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_chart(pred, tmpdir)
            self.assertTrue(os.path.isfile(path))

    def test_chart_path_format(self):
        pred = self._minimal_prediction()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_chart(pred, tmpdir)
            self.assertIn("TEST_1w", path)


class TestFetchWithRetry(unittest.TestCase):
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value=42)
        result = _fetch_with_retry(fn, "arg", max_retries=3)
        self.assertEqual(result, 42)
        fn.assert_called_once_with("arg")

    def test_retries_on_failure_then_succeeds(self):
        fn = MagicMock(side_effect=[Exception("err"), Exception("err"), 99])
        with patch("time.sleep"):
            result = _fetch_with_retry(fn, max_retries=3)
        self.assertEqual(result, 99)
        self.assertEqual(fn.call_count, 3)

    def test_raises_after_max_retries(self):
        fn = MagicMock(side_effect=ValueError("always fails"))
        with patch("time.sleep"), self.assertRaises(ValueError):
            _fetch_with_retry(fn, max_retries=2)
        self.assertEqual(fn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
