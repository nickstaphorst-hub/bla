"""Unit tests for the pure analysis helpers of the crypto market scan.

These tests intentionally use only the standard library, so they run without
the Workflows SDK installed (``python3 -m unittest test_crypto_analysis.py``)
and also under the project's pytest setup via ``uv``.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "worker" / "workflows"))

import crypto_analysis  # noqa: E402


class TestIndicators(unittest.TestCase):
    def test_sma_basic(self) -> None:
        self.assertEqual(crypto_analysis.sma([1, 2, 3, 4, 5], 5), 3.0)
        self.assertEqual(crypto_analysis.sma([1, 2, 3, 4, 5], 2), 4.5)

    def test_sma_needs_enough_data(self) -> None:
        self.assertIsNone(crypto_analysis.sma([1, 2], 5))
        self.assertIsNone(crypto_analysis.sma([], 3))
        self.assertIsNone(crypto_analysis.sma([1, 2, 3], 0))

    def test_rsi_all_gains_is_100(self) -> None:
        closes = [float(i) for i in range(1, 25)]
        self.assertEqual(crypto_analysis.rsi(closes, 14), 100.0)

    def test_rsi_all_losses_is_0(self) -> None:
        closes = [float(i) for i in range(25, 1, -1)]
        self.assertEqual(crypto_analysis.rsi(closes, 14), 0.0)

    def test_rsi_known_value(self) -> None:
        # diffs over the last 3 candles: +1, +1, -1 -> avg_gain 2/3, avg_loss 1/3
        # rs = 2 -> rsi = 100 - 100/3 = 66.666...
        self.assertAlmostEqual(crypto_analysis.rsi([10, 11, 12, 13, 12], 3), 200.0 / 3.0)

    def test_rsi_too_short_is_none(self) -> None:
        self.assertIsNone(crypto_analysis.rsi([1.0, 2.0], 14))

    def test_rsi_flat_is_50(self) -> None:
        self.assertEqual(crypto_analysis.rsi([5.0] * 15, 14), 50.0)

    def test_volatility_flat_is_zero(self) -> None:
        self.assertEqual(crypto_analysis.volatility_pct([5.0] * 40), 0.0)

    def test_volatility_short_is_none(self) -> None:
        self.assertIsNone(crypto_analysis.volatility_pct([5.0, 5.0]))

    def test_volatility_positive_for_moving_series(self) -> None:
        closes = [100.0 * (1.01**i) for i in range(40)]
        vol = crypto_analysis.volatility_pct(closes)
        self.assertIsNotNone(vol)
        self.assertGreater(vol, 0.0)

    def test_classify_trend(self) -> None:
        self.assertEqual(crypto_analysis.classify_trend(110.0, 105.0, 100.0), "bullish")
        self.assertEqual(crypto_analysis.classify_trend(90.0, 95.0, 100.0), "bearish")
        self.assertEqual(crypto_analysis.classify_trend(105.0, 100.0, 102.0), "neutral")
        self.assertEqual(crypto_analysis.classify_trend(110.0, None, None), "neutral")


class TestParsing(unittest.TestCase):
    def test_parse_binance_klines(self) -> None:
        rows = [
            [1700000000000, "100", "110", "99", "105", "12.5", 1700003600000, "1312"],
            [1700003600000, "105", "106", "101", "104", "10.0", 1700007200000, "1040"],
        ]
        parsed = crypto_analysis.parse_binance_klines(rows)
        self.assertEqual(parsed["closes"], [105.0, 104.0])
        self.assertEqual(parsed["volumes"], [12.5, 10.0])

    def test_parse_binance_klines_skips_malformed(self) -> None:
        parsed = crypto_analysis.parse_binance_klines([["bad"], [1, "a", "b", "c", "100", "1"]])
        self.assertEqual(parsed["closes"], [100.0])

    def test_latest_timestamp(self) -> None:
        coins = [
            {"last_updated": "2026-11-14T10:00:00Z"},
            {"last_updated": "2026-11-14T11:30:00Z"},
            {},
        ]
        self.assertEqual(crypto_analysis.latest_timestamp(coins), "2026-11-14T11:30:00Z")
        self.assertEqual(crypto_analysis.latest_timestamp([]), "onbekend")


class TestSnapshotRow(unittest.TestCase):
    COIN = {
        "symbol": "btc",
        "name": "Bitcoin",
        "market_cap_rank": 1,
        "current_price": 65000.0,
        "market_cap": 1_300_000_000_000,
        "total_volume": 25_000_000_000,
        "price_change_percentage_24h_in_currency": -1.25,
        "price_change_percentage_7d_in_currency": 4.5,
        "price_change_percentage_30d_in_currency": 12.0,
        "last_updated": "2026-11-14T11:30:00Z",
    }

    def test_row_with_klines(self) -> None:
        rising = [60_000.0 + i * 100.0 for i in range(60)]
        row = crypto_analysis.build_snapshot_row(self.COIN, {"closes": rising, "volumes": []})
        self.assertEqual(row["symbol"], "BTC")
        self.assertEqual(row["klines_source"], "binance")
        self.assertEqual(row["trend"], "bullish")
        self.assertIsNotNone(row["rsi_14"])
        self.assertIsNotNone(row["sma_20"])
        self.assertIsNotNone(row["sma_50"])
        self.assertEqual(row["change_24h_pct"], -1.25)

    def test_row_without_klines(self) -> None:
        row = crypto_analysis.build_snapshot_row(self.COIN, None)
        self.assertEqual(row["klines_source"], "none")
        self.assertIsNone(row["rsi_14"])
        self.assertIsNone(row["sma_20"])
        self.assertEqual(row["trend"], "neutral")
        self.assertEqual(row["price"], 65000.0)


class TestPromptAndReport(unittest.TestCase):
    ROWS = [
        crypto_analysis.build_snapshot_row(
            {
                "symbol": "btc",
                "name": "Bitcoin",
                "market_cap_rank": 1,
                "current_price": 65000.0,
                "market_cap": 1.3e12,
                "total_volume": 2.5e10,
                "price_change_percentage_24h_in_currency": 1.5,
                "price_change_percentage_7d_in_currency": 3.0,
                "price_change_percentage_30d_in_currency": 5.0,
            },
            None,
        ),
        crypto_analysis.build_snapshot_row(
            {
                "symbol": "eth",
                "name": "Ethereum",
                "market_cap_rank": 2,
                "current_price": 3200.0,
                "market_cap": 3.8e11,
                "total_volume": 1.1e10,
                "price_change_percentage_24h_in_currency": -0.4,
                "price_change_percentage_7d_in_currency": 2.2,
                "price_change_percentage_30d_in_currency": -1.0,
            },
            None,
        ),
    ]

    PROPOSAL = {
        "market_regime": "risico-on",
        "regime_rationale": "Breed positief momentum.",
        "narrative": "De markt trekt aan, met Bitcoin als motor.",
        "trade_ideas": [
            {
                "symbol": "BTC",
                "direction": "long",
                "conviction_pct": 72,
                "entry_low": 64000.0,
                "entry_high": 65500.0,
                "stop_loss": 61000.0,
                "take_profit_1": 68000.0,
                "take_profit_2": 70000.0,
                "timeframe": "24-72 uur",
                "rationale": "Trend boven SMA20/50.",
            }
        ],
        "key_risks": ["Renderings van macro-data", "Lage liquiditeit in het weekend"],
    }

    def test_prompt_contains_rules_and_data(self) -> None:
        prompt = crypto_analysis.build_analysis_prompt(self.ROWS, "balanced", 3, "nl")
        self.assertIn("balanced", prompt)
        self.assertIn("'nl'", prompt)
        # embedded data must be valid JSON
        json_part = prompt.split("DATA (JSON, candle-interval 4u):\n")[1].split("\n\nREGELS")[0]
        data = json.loads(json_part)
        self.assertEqual(data[0]["symbol"], "BTC")
        self.assertLessEqual(len(data), 2)

    def test_compact_rows_valid_json(self) -> None:
        data = json.loads(crypto_analysis.compact_rows_for_prompt(self.ROWS))
        self.assertEqual(len(data), 2)
        self.assertNotIn("klines_source", data[0])

    def test_report_renders_proposal(self) -> None:
        report = crypto_analysis.render_report(
            as_of="2026-11-14T11:30:00Z",
            quote_currency="usd",
            risk_profile="balanced",
            rows=self.ROWS,
            proposal=self.PROPOSAL,
            model="mistral-medium-latest",
        )
        self.assertIn("# Crypto Marktscan", report)
        self.assertIn("### 1. BTC - LONG", report)
        self.assertIn("Entry-zone", report)
        self.assertIn("64,000.00", report)
        self.assertIn("Ethereum (ETH)", report)
        self.assertIn("Disclaimer", report)

    def test_report_handles_no_ideas(self) -> None:
        proposal = dict(self.PROPOSAL, trade_ideas=[])
        report = crypto_analysis.render_report(
            "t", "usd", "balanced", self.ROWS, proposal, "m"
        )
        self.assertIn("Geen voorstellen deze scan.", report)


if __name__ == "__main__":
    unittest.main()
