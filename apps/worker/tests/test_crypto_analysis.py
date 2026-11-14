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


RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Feed</title>
<item>
  <title>Bitcoin &lt;b&gt;surges&lt;/b&gt;   to a record high</title>
  <link>https://example.com/btc-surge</link>
  <pubDate>Sat, 14 Nov 2026 11:28:12 +0000</pubDate>
</item>
<item>
  <title>Crypto prices crash amid massive liquidations</title>
  <link>https://example.com/crash</link>
  <pubDate>Sat, 14 Nov 2026 09:00:00 +0000</pubDate>
</item>
<item>
  <title></title>
  <link>https://example.com/empty</link>
</item>
</channel></rss>
"""


class TestFearGreed(unittest.TestCase):
    def test_parse_full_payload(self) -> None:
        payload = {
            "data": [
                {"value": "13", "value_classification": "Extreme Fear"},
                {"value": "20", "value_classification": "Extreme Fear"},
                {"value": "28", "value_classification": "Fear"},
            ]
        }
        fg = crypto_analysis.parse_fear_greed(payload)
        self.assertEqual(fg["value"], 13)
        self.assertEqual(fg["classification"], "Extreme Fear")
        self.assertEqual(fg["value_yesterday"], 20)
        self.assertIsNone(fg["value_last_week"])
        self.assertEqual(fg["change_1d"], -7)
        self.assertIsNone(fg["change_7d"])
        self.assertEqual(fg["history"], [13, 20, 28])

    def test_parse_with_week_history(self) -> None:
        payload = {"data": [{"value": "55", "value_classification": "Greed"}]}
        for day in range(1, 8):
            payload["data"].append({"value": str(50 + day), "value_classification": "x"})
        fg = crypto_analysis.parse_fear_greed(payload)
        self.assertEqual(fg["value_last_week"], 57)
        self.assertEqual(fg["change_7d"], -2)

    def test_parse_bad_payloads(self) -> None:
        self.assertIsNone(crypto_analysis.parse_fear_greed(None))
        self.assertIsNone(crypto_analysis.parse_fear_greed({}))
        self.assertIsNone(crypto_analysis.parse_fear_greed({"data": []}))
        self.assertIsNone(crypto_analysis.parse_fear_greed({"data": [{"value": "x"}]}))
        self.assertIsNone(crypto_analysis.parse_fear_greed({"data": [{}]}))


class TestNewsParsing(unittest.TestCase):
    def test_parse_rss_feed(self) -> None:
        items = crypto_analysis.parse_rss_feed(RSS_SAMPLE, "testfeed")
        self.assertEqual(len(items), 2)  # empty title item is skipped
        self.assertEqual(items[0]["title"], "Bitcoin surges to a record high")
        self.assertEqual(items[0]["source"], "testfeed")
        self.assertEqual(items[0]["url"], "https://example.com/btc-surge")
        self.assertTrue(items[0]["published"].startswith("2026-11-14T11:28:12"))

    def test_parse_rss_feed_invalid_xml(self) -> None:
        self.assertEqual(crypto_analysis.parse_rss_feed("<not-rss", "s"), [])
        self.assertEqual(crypto_analysis.parse_rss_feed("", "s"), [])

    def test_strip_html(self) -> None:
        self.assertEqual(crypto_analysis.strip_html("<p>Hi &amp; welcome</p>\n\n  there"), "Hi & welcome there")

    def test_score_positive_negative_neutral(self) -> None:
        self.assertGreater(crypto_analysis.score_headline_sentiment("Bitcoin surges to a record high"), 0)
        self.assertLess(crypto_analysis.score_headline_sentiment("Crypto prices crash amid massive liquidations"), 0)
        self.assertEqual(crypto_analysis.score_headline_sentiment("Weekly roundup of crypto markets"), 0)

    def test_score_is_clamped(self) -> None:
        title = "crash plunge slump bearish hack fraud dump liquidation drop fall warn"
        self.assertEqual(crypto_analysis.score_headline_sentiment(title), -2)

    def test_score_deterministic_no_stem_false_positives(self) -> None:
        # 'bank' must not trigger the 'ban' keyword, 'banking' neither
        self.assertEqual(crypto_analysis.score_headline_sentiment("New crypto bank opens branch in Europe"), 0)

    def test_merge_news_dedupes_and_annotates(self) -> None:
        feed_a = crypto_analysis.parse_rss_feed(RSS_SAMPLE, "a")
        feed_b = [
            {
                "title": "Bitcoin surges to a record high",  # duplicate of feed_a
                "url": "https://other.example/x",
                "published": "2026-11-14T11:30:00+00:00",
                "source": "b",
            },
            {
                "title": "Ethereum adoption grows after upgrade approval",
                "url": "https://other.example/y",
                "published": "2026-11-14T10:00:00+00:00",
                "source": "b",
            },
        ]
        merged = crypto_analysis.merge_news({"a": feed_a, "b": feed_b}, max_items=10)
        titles = [it["title"] for it in merged]
        self.assertEqual(len(titles), len(set(titles)))  # no duplicates
        self.assertEqual(len(merged), 3)
        # sorted newest first
        self.assertEqual(merged[0]["title"], "Bitcoin surges to a record high")
        # annotated with sentiment
        for item in merged:
            self.assertIn("sentiment_score", item)
            self.assertIn(item["sentiment"], ("positief", "negatief", "neutraal"))

    def test_merge_news_trims_to_max_items(self) -> None:
        feed = [
            {
                "title": f"Headline number {i} about markets",
                "url": "",
                "published": f"2026-11-14T0{i}:00:00+00:00" if i < 10 else "2026-11-14T10:00:00+00:00",
                "source": "s",
            }
            for i in range(12)
        ]
        merged = crypto_analysis.merge_news({"s": feed}, max_items=5)
        self.assertEqual(len(merged), 5)

    def test_merge_news_lookback_filters_old_items(self) -> None:
        feed = [
            {"title": "Fresh headline", "url": "", "published": "2026-11-14T11:00:00+00:00", "source": "s"},
            {"title": "Ancient headline", "url": "", "published": "2026-10-01T11:00:00+00:00", "source": "s"},
        ]
        merged = crypto_analysis.merge_news({"s": feed}, max_items=10, lookback_hours=48)
        titles = [it["title"] for it in merged]
        self.assertIn("Fresh headline", titles)
        self.assertNotIn("Ancient headline", titles)

    def test_merge_news_empty(self) -> None:
        self.assertEqual(crypto_analysis.merge_news({}, 10), [])
        self.assertEqual(crypto_analysis.merge_news({"s": []}, 10), [])

    def test_summarize_news_sentiment(self) -> None:
        news = [
            {"sentiment_score": 2},
            {"sentiment_score": 1},
            {"sentiment_score": 0},
            {"sentiment_score": -2},
        ]
        summary = crypto_analysis.summarize_news_sentiment(news)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["positive"], 2)
        self.assertEqual(summary["negative"], 1)
        self.assertEqual(summary["neutral"], 1)
        self.assertEqual(summary["average_score"], 0.25)
        self.assertEqual(summary["label"], "neutraal")

    def test_summarize_news_sentiment_empty(self) -> None:
        self.assertIsNone(crypto_analysis.summarize_news_sentiment([]))

    def test_coin_mention_counts(self) -> None:
        rows = [
            {"symbol": "BTC", "name": "Bitcoin"},
            {"symbol": "ETH", "name": "Ethereum"},
            {"symbol": "SOL", "name": "Solana"},
            {"symbol": "X", "name": "X"},  # too short: ignored
        ]
        news = [
            {"title": "Bitcoin surges to a record high"},
            {"title": "Bitcoin and Solana lead the recovery"},
            {"title": "Nothing to see here"},
        ]
        counts = crypto_analysis.coin_mention_counts(news, rows)
        by_symbol = {c["symbol"]: c["count"] for c in counts}
        self.assertEqual(by_symbol.get("BTC"), 2)
        self.assertEqual(by_symbol.get("SOL"), 1)
        self.assertNotIn("ETH", by_symbol)
        self.assertNotIn("X", by_symbol)
        # sorted by count desc
        self.assertEqual(counts[0]["symbol"], "BTC")

    def test_coin_mention_counts_word_boundary(self) -> None:
        # 'sol' inside 'solely' must not count as a Solana mention
        news = [{"title": "Analysts solely focus on macro data"}]
        counts = crypto_analysis.coin_mention_counts(news, [{"symbol": "SOL", "name": "Solana"}])
        self.assertEqual(counts, [])

    def test_coin_mention_counts_no_news(self) -> None:
        self.assertEqual(crypto_analysis.coin_mention_counts([], [{"symbol": "BTC", "name": "Bitcoin"}]), [])


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

    FEAR_GREED = {
        "value": 13,
        "classification": "Extreme Fear",
        "value_yesterday": 20,
        "value_last_week": 28,
        "change_1d": -7,
        "change_7d": -15,
        "history": [13, 20, 28],
    }

    NEWS = [
        {
            "title": "Bitcoin surges to a record high",
            "source": "cointelegraph",
            "published": "2026-11-14T11:28:12+00:00",
            "url": "https://example.com/btc",
            "sentiment_score": 1,
            "sentiment": "positief",
        },
        {
            "title": "Crypto prices crash amid massive liquidations",
            "source": "decrypt",
            "published": "2026-11-14T09:00:00+00:00",
            "url": "https://example.com/crash",
            "sentiment_score": -2,
            "sentiment": "negatief",
        },
    ]

    NEWS_SENTIMENT = {
        "average_score": -0.5,
        "label": "negatief",
        "positive": 1,
        "negative": 1,
        "neutral": 0,
        "total": 2,
    }

    COIN_MENTIONS = [{"symbol": "BTC", "name": "Bitcoin", "count": 1}]

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
        json_part = (
            prompt.split("PRIJSDATA (JSON, candle-interval 4u, top 2 op market cap):\n")[1]
            .split("\n\nMARKT-SENTIMENT")[0]
        )
        data = json.loads(json_part)
        self.assertEqual(data[0]["symbol"], "BTC")
        self.assertLessEqual(len(data), 2)
        # without sentiment the prompt says so instead of fabricating data
        self.assertIn("niet beschikbaar", prompt)

    def test_prompt_with_sentiment_and_news(self) -> None:
        prompt = crypto_analysis.build_analysis_prompt(
            self.ROWS,
            "balanced",
            3,
            "nl",
            fear_greed=self.FEAR_GREED,
            news=self.NEWS,
            news_sentiment=self.NEWS_SENTIMENT,
            coin_mentions=self.COIN_MENTIONS,
        )
        self.assertIn("Fear & Greed", prompt)
        self.assertIn("Extreme Fear", prompt)
        self.assertIn("record high", prompt)
        self.assertIn("sentiment_score", prompt)
        self.assertIn("MUNTVERMELDINGEN", prompt)
        self.assertIn("contextsignalen", prompt)
        # news block must be valid JSON
        news_part = prompt.split("deterministische keyword-score -2..+2 als ruw signaal):\n")[1]
        news_data = json.loads(news_part.split("\n\nNIEUWS-SAMENVATTING")[0])
        self.assertEqual(len(news_data), 2)
        self.assertEqual(news_data[0]["sentiment"], "positief")

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
        # no sentiment passed: sections must be omitted entirely
        self.assertNotIn("## Sentiment", report)
        self.assertNotIn("## Nieuws", report)

    def test_report_with_sentiment_and_news(self) -> None:
        report = crypto_analysis.render_report(
            as_of="2026-11-14T11:30:00Z",
            quote_currency="usd",
            risk_profile="balanced",
            rows=self.ROWS,
            proposal=self.PROPOSAL,
            model="mistral-medium-latest",
            fear_greed=self.FEAR_GREED,
            news=self.NEWS,
            news_sentiment=self.NEWS_SENTIMENT,
            coin_mentions=self.COIN_MENTIONS,
        )
        self.assertIn("## Sentiment", report)
        self.assertIn("Fear & Greed Index:** 13/100", report)
        self.assertIn("24u eerder: 20 (-7)", report)
        self.assertIn("Nieuwssentiment:** negatief", report)
        self.assertIn("Meest genoemd in het nieuws:** BTC (1x)", report)
        self.assertIn("## Nieuws", report)
        self.assertIn("record high", report)
        self.assertIn("crash amid massive", report)
        self.assertIn("negatief (-2)", report)

    def test_report_with_news_but_no_fear_greed(self) -> None:
        report = crypto_analysis.render_report(
            "t", "usd", "balanced", self.ROWS, self.PROPOSAL, "m",
            fear_greed=None,
            news=self.NEWS,
            news_sentiment=self.NEWS_SENTIMENT,
            coin_mentions=[],
        )
        self.assertIn("## Sentiment", report)
        self.assertIn("niet beschikbaar", report)
        self.assertIn("Geen van de top-munten staat in de koppen.", report)

    def test_prompt_spot_mode_adds_rules(self) -> None:
        prompt = crypto_analysis.build_analysis_prompt(
            self.ROWS, "balanced", 3, "nl", trading_mode="spot"
        )
        self.assertIn("HANDELSMODUS: SPOT-ONLY", prompt)
        self.assertIn("ALLEEN direction 'long' of 'none'", prompt)
        self.assertIn("KOOPZONE", prompt)
        self.assertIn("INVALIDATIENIVEAU", prompt)
        self.assertIn("VERKOOPDOELEN", prompt)
        self.assertIn("niet uitvoerbaar", prompt)

    def test_prompt_default_mode_has_no_spot_rules(self) -> None:
        prompt = crypto_analysis.build_analysis_prompt(self.ROWS, "balanced", 3, "nl")
        self.assertNotIn("HANDELSMODUS", prompt)
        self.assertNotIn("SPOT-ONLY", prompt)

    def test_report_spot_mode_labels(self) -> None:
        report = crypto_analysis.render_report(
            "2026-11-14T11:30:00Z",
            "usd",
            "balanced",
            self.ROWS,
            self.PROPOSAL,
            "mistral-medium-latest",
            trading_mode="spot",
        )
        self.assertIn("Handelsmodus: spot (alleen koop, geen hefboom)", report)
        self.assertIn("Handelsmodus spot: alleen koopideeën", report)
        self.assertIn("**Koopzone:**", report)
        self.assertIn("**Invalidatie (verkoop onder):**", report)
        self.assertIn("**Verkoopdoelen:**", report)
        # default labels must not appear in spot mode
        self.assertNotIn("**Entry-zone:**", report)
        self.assertNotIn("**Stop-loss:**", report)
        self.assertNotIn("**Take-profit:**", report)

    def test_report_spot_mode_marks_short_not_executable(self) -> None:
        proposal = dict(self.PROPOSAL)
        proposal["trade_ideas"] = [
            {
                "symbol": "BTC",
                "direction": "short",
                "conviction_pct": 60,
                "entry_low": 68000.0,
                "entry_high": 69000.0,
                "stop_loss": 71000.0,
                "take_profit_1": 64000.0,
                "take_profit_2": None,
                "timeframe": "24-72 uur",
                "rationale": "Bearish trend onder SMA20/50.",
            }
        ]
        report = crypto_analysis.render_report(
            "t", "usd", "balanced", self.ROWS, proposal, "m", trading_mode="spot"
        )
        self.assertIn("op de spotmarkt niet uitvoerbaar", report)
        self.assertNotIn("**Koopzone:**", report)  # short renders no buy levels

    def test_report_default_mode_labels_unchanged(self) -> None:
        report = crypto_analysis.render_report(
            "t", "usd", "balanced", self.ROWS, self.PROPOSAL, "m", trading_mode="any"
        )
        self.assertIn("**Entry-zone:**", report)
        self.assertIn("**Stop-loss:**", report)
        self.assertIn("**Take-profit:**", report)
        self.assertNotIn("Handelsmodus", report)

    def test_report_handles_no_ideas(self) -> None:
        proposal = dict(self.PROPOSAL, trade_ideas=[])
        report = crypto_analysis.render_report(
            "t", "usd", "balanced", self.ROWS, proposal, "m"
        )
        self.assertIn("Geen voorstellen deze scan.", report)


if __name__ == "__main__":
    unittest.main()
