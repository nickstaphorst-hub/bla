"""Pure analysis helpers for the crypto market scan workflow.

This module deliberately depends only on the Python standard library: the
functions are deterministic and side-effect free, so they can run inside the
workflow body (the determinism sandbox) and be unit-tested without the
Workflows SDK installed. All network I/O lives in ``crypto_scan.py``.

Sentiment & news: the workflow blends technical price data with market-wide
sentiment (the Fear & Greed Index of alternative.me) and recent crypto news
headlines from public RSS feeds. Headlines get a deterministic keyword score
(-2..+2) so the signal is reproducible; the LLM weighs them, it does not
generate them.
"""

from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

# Top fiat-pegged coins: trading them against USDT is pointless and their
# TA values are noise, so the workflow filters them out before analysis.
STABLE_SYMBOLS = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USDS"}

MAX_TOKENS_ROWS = 30  # rows kept for the model prompt, hard safety bound
MAX_NEWS_TITLE_LEN = 200  # hard bound for one headline in the prompt/report

# Free public RSS feeds (no API key). Fetched in an activity; parsed here.
NEWS_FEEDS: dict[str, str] = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
}

# Deterministic keyword lexicon for headline sentiment. Exact token matches
# only, all forms spelled out: no stemming surprises in tests or production.
_SENTIMENT_POSITIVE = frozenset(
    {
        "surge", "surges", "surged", "surging",
        "soar", "soars", "soared", "soaring",
        "rally", "rallies", "rallied", "rallying",
        "bullish", "bull",
        "record", "highs", "ath",
        "gain", "gains", "gained", "gaining",
        "jump", "jumps", "jumped", "jumping",
        "boost", "boosts", "boosted", "boosting",
        "adoption", "adopts", "adopt",
        "approve", "approves", "approved", "approval",
        "inflow", "inflows",
        "upgrade", "upgrades", "upgraded",
        "partnership", "partnerships",
        "buy", "buys", "buying", "accumulation",
        "breakout", "breakouts",
        "recovery", "recovers", "recovered",
        "rise", "rises", "rising", "rose",
        "climb", "climbs", "climbing", "climbed",
        "uptrend", "green",
    }
)
_SENTIMENT_NEGATIVE = frozenset(
    {
        "crash", "crashes", "crashed", "crashing",
        "plunge", "plunges", "plunged", "plunging",
        "tumble", "tumbles", "tumbled", "tumbling",
        "slump", "slumps", "slumped", "slumping",
        "bearish", "bear",
        "loss", "losses",
        "hack", "hacks", "hacked", "hacking",
        "exploit", "exploits", "exploited",
        "breach", "breaches", "breached",
        "theft", "stolen",
        "sell-off", "selloff", "sell", "sells", "selling",
        "ban", "bans", "banned",
        "lawsuit", "sues", "sued",
        "fraud", "scam", "scams",
        "dump", "dumps", "dumped", "dumping",
        "fear", "fears",
        "liquidation", "liquidations",
        "outflow", "outflows",
        "warn", "warns", "warning", "warnings",
        "decline", "declines", "declined", "declining",
        "drop", "drops", "dropped", "dropping",
        "fall", "falls", "fell", "falling",
        "downtrend", "risk-off",
    }
)


# --------------------------------------------------------------------------
# Technical indicators (price data)
# --------------------------------------------------------------------------
def sma(values: list[float], period: int) -> float | None:
    """Simple moving average over the last ``period`` values."""
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period

def rsi(closes: list[float], period: int = 14) -> float | None:
    """Cutler's RSI (SMA-based, fully deterministic on the same input)."""
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for prev, cur in zip(closes[-(period + 1) : -1], closes[-period:], strict=False):
        diff = cur - prev
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if not gains:  # pragma: no cover - length guaranteed above
        return None
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def volatility_pct(closes: list[float], lookback: int = 30) -> float | None:
    """Stddev of candle-to-candle percentage returns, over the last ``lookback`` candles."""
    tail = closes[-(lookback + 1) :] if lookback < len(closes) else closes
    if len(tail) < 3:
        return None
    returns: list[float] = []
    for prev, cur in zip(tail, tail[1:], strict=False):
        if prev == 0:
            continue
        returns.append((cur - prev) / prev * 100.0)
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var)

def classify_trend(price: float, sma_20: float | None, sma_50: float | None) -> str:
    """Coarse trend label from price vs SMA20 and SMA50."""
    if sma_20 is None or sma_50 is None:
        return "neutral"
    if price > sma_20 > sma_50:
        return "bullish"
    if price < sma_20 < sma_50:
        return "bearish"
    return "neutral"

def parse_binance_klines(rows: list[list[Any]]) -> dict[str, Any]:
    """Reduce Binance klines rows to the series the indicators need.

    Binance rows are ``[openTime, open, high, low, close, volume, closeTime,
    quoteVolume, ...]`` with OHLCV as strings.
    """
    closes: list[float] = []
    volumes: list[float] = []
    for row in rows:
        if len(row) < 6:  # malformed row, skip rather than crash the scan
            continue
        try:
            closes.append(float(row[4]))
            volumes.append(float(row[5]))
        except (TypeError, ValueError):
            continue
    return {"closes": closes, "volumes": volumes}

def build_snapshot_row(
    coin: dict[str, Any],
    kline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge one CoinGecko market row with parsed klines into a snapshot row.

    ``kline`` is ``None`` when Binance has no pair for the coin or the fetch
    failed; the row then keeps CoinGecko's percentage changes without TA.
    """
    symbol = str(coin.get("symbol", "")).upper()
    closes: list[float] = list(kline.get("closes", [])) if kline else []
    price = float(coin.get("current_price") or 0.0)
    sma_20 = sma(closes, 20)
    sma_50 = sma(closes, 50)
    return {
        "symbol": symbol,
        "name": str(coin.get("name", symbol)),
        "rank": int(coin.get("market_cap_rank") or 0),
        "price": price,
        "market_cap": float(coin.get("market_cap") or 0.0),
        "volume_24h": float(coin.get("total_volume") or 0.0),
        "change_24h_pct": coin.get("price_change_percentage_24h_in_currency"),
        "change_7d_pct": coin.get("price_change_percentage_7d_in_currency"),
        "change_30d_pct": coin.get("price_change_percentage_30d_in_currency"),
        "rsi_14": rsi(closes, 14),
        "sma_20": sma_20,
        "sma_50": sma_50,
        "volatility_pct": volatility_pct(closes),
        "trend": classify_trend(price, sma_20, sma_50),
        "klines_source": "binance" if closes else "none",
    }

def latest_timestamp(coins: list[dict[str, Any]]) -> str:
    """Newest ``last_updated`` across CoinGecko rows, as an ISO-ish string."""
    stamps = [c.get("last_updated") for c in coins if c.get("last_updated")]
    return max(stamps) if stamps else "onbekend"


# --------------------------------------------------------------------------
# Sentiment: Fear & Greed Index (alternative.me)
# --------------------------------------------------------------------------

def parse_fear_greed(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the alternative.me Fear & Greed response.

    The API returns ``{"data": [{"value": "13", "value_classification":
    "Extreme Fear", "timestamp": "..."}]}`` with the newest entry first and
    (with ``limit``) up to N days of history. Returns ``None`` when the
    payload holds no usable current value; sentiment is a bonus signal, the
    scan must survive without it.
    """
    data = (payload or {}).get("data") or []
    values: list[int] = []
    for entry in data:
        try:
            values.append(int(str(entry.get("value")).strip()))
        except (TypeError, ValueError):
            values.append(-1)  # keep positional alignment with data entries
    if not data or values[0] < 0:
        return None

    def _value_at(index: int) -> int | None:
        if index >= len(values) or values[index] < 0:
            return None
        return values[index]

    value = values[0]
    yesterday = _value_at(1)
    last_week = _value_at(7)
    return {
        "value": value,
        "classification": str(data[0].get("value_classification", "onbekend")),
        "value_yesterday": yesterday,
        "value_last_week": last_week,
        "change_1d": value - yesterday if yesterday is not None else None,
        "change_7d": value - last_week if last_week is not None else None,
        "history": [v for v in values if v >= 0],
    }


# --------------------------------------------------------------------------
# Sentiment: news headlines (public RSS feeds)
# --------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove HTML tags/entities and collapse whitespace in feed text."""
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = html.unescape(cleaned)
    return " ".join(cleaned.split())

def score_headline_sentiment(title: str) -> int:
    """Deterministic keyword score for one headline, clamped to [-2, +2].

    Exact-token matching against a fixed lexicon: same headline, same score,
    in tests and in production. The score is a coarse signal for the model,
    not a truth claim about the article.
    """
    tokens = re.findall(r"[a-z0-9'-]+", (title or "").lower())
    score = 0
    for token in tokens:
        if token in _SENTIMENT_POSITIVE:
            score += 1
        elif token in _SENTIMENT_NEGATIVE:
            score -= 1
    return max(-2, min(2, score))

def sentiment_label(score: int) -> str:
    """Dutch label for a headline score."""
    if score > 0:
        return "positief"
    if score < 0:
        return "negatief"
    return "neutraal"

def parse_rss_feed(xml_text: str, source: str) -> list[dict[str, Any]]:
    """Parse one RSS 2.0 feed body into plain news items.

    Items carry ``title``, ``url``, ``published`` (ISO 8601 or ``None``) and
    ``source``. Malformed XML or missing fields never raise: a feed that
    cannot be parsed simply contributes no headlines.
    """
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return []
    items: list[dict[str, Any]] = []
    for element in root.findall(".//item"):
        raw_title = strip_html(element.findtext("title") or "")
        if not raw_title:
            continue
        url = (element.findtext("link") or element.findtext("guid") or "").strip()
        published: str | None = None
        try:
            published = parsedate_to_datetime(
                (element.findtext("pubDate") or "").strip()
            ).isoformat()
        except (TypeError, ValueError):
            published = None
        items.append(
            {
                "title": raw_title[:MAX_NEWS_TITLE_LEN],
                "url": url,
                "published": published,
                "source": source,
            }
        )
    return items

def merge_news(
    feeds: dict[str, list[dict[str, Any]]],
    max_items: int = 12,
    lookback_hours: int = 48,
) -> list[dict[str, Any]]:
    """Deduplicate, age-filter, sort and score the parsed feeds.

    Deterministic by construction: no wall-clock access. The reference moment
    is the newest headline in the merged set, so the same input always yields
    the same selection. Items without a parsable date are kept (they cannot
    be judged old). Output items are annotated with ``sentiment_score`` and
    ``sentiment``.
    """
    merged: list[dict[str, Any]] = []
    for items in feeds.values():
        merged.extend(items)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in merged:
        key = re.sub(r"[^a-z0-9]+", "", item["title"].lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique.sort(key=lambda it: it.get("published") or "", reverse=True)

    newest_iso = next((it.get("published") for it in unique if it.get("published")), None)
    if newest_iso is not None:
        try:
            cutoff = datetime.fromisoformat(newest_iso) - timedelta(hours=lookback_hours)
            age_ok: list[dict[str, Any]] = []
            for item in unique:
                iso = item.get("published")
                if not iso:
                    age_ok.append(item)
                    continue
                if datetime.fromisoformat(iso) >= cutoff:
                    age_ok.append(item)
            unique = age_ok
        except ValueError:  # pragma: no cover - parse_rss_feed guarantees ISO
            pass

    result: list[dict[str, Any]] = []
    for item in unique[:max(1, max_items)]:
        score = score_headline_sentiment(item["title"])
        result.append(
            {
                "title": item["title"],
                "source": item.get("source", "onbekend"),
                "published": item.get("published"),
                "url": item.get("url", ""),
                "sentiment_score": score,
                "sentiment": sentiment_label(score),
            }
        )
    return result

def summarize_news_sentiment(news: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate the per-headline scores into one summary block."""
    if not news:
        return None
    scores = [int(item.get("sentiment_score", 0)) for item in news]
    positive = sum(1 for s in scores if s > 0)
    negative = sum(1 for s in scores if s < 0)
    neutral = len(scores) - positive - negative
    avg = round(sum(scores) / len(scores), 2)
    if avg >= 0.5:
        label = "positief"
    elif avg <= -0.5:
        label = "negatief"
    else:
        label = "neutraal"
    return {
        "average_score": avg,
        "label": label,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "total": len(scores),
    }

def coin_mention_counts(
    news: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Count which of the snapshot coins appear in the headlines.

    Word-boundary matching on symbol and name, case-insensitive; short keys
    (under 3 characters) are skipped to avoid noise like 'X' or 'PEPE'-style
    false hits inside other words.
    """
    if not news:
        return []
    titles = [item.get("title", "").lower() for item in news]
    counts: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        name = str(row.get("name", ""))
        patterns = []
        for key in (symbol.lower(), name.lower()):
            if len(key) >= 3 and key not in patterns:
                patterns.append(re.escape(key))
        if not patterns:
            continue
        # Whole-word match with a plural allowance; prevents false hits like
        # 'sol' inside 'solely' while still counting 'bitcoins'.
        regex = re.compile(r"\b(?:" + "|".join(patterns) + r")(?:s|es)?\b")
        count = sum(1 for title in titles if regex.search(title))
        if count > 0:
            counts.append({"symbol": symbol, "name": name, "count": count})
    counts.sort(key=lambda c: (-c["count"], c["symbol"]))
    return counts


# --------------------------------------------------------------------------
# Model prompt and Markdown report
# --------------------------------------------------------------------------

def _round_or_none(value: float | int | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)

def compact_rows_for_prompt(rows: list[dict[str, Any]]) -> str:
    """JSON rendering of the snapshot, trimmed to what the model needs."""
    compact = []
    for row in rows[:MAX_TOKENS_ROWS]:
        compact.append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "price": row["price"],
                "market_cap": _round_or_none(row["market_cap"], 0),
                "volume_24h": _round_or_none(row["volume_24h"], 0),
                "change_24h_pct": _round_or_none(row["change_24h_pct"], 2),
                "change_7d_pct": _round_or_none(row["change_7d_pct"], 2),
                "change_30d_pct": _round_or_none(row["change_30d_pct"], 2),
                "rsi_14": _round_or_none(row["rsi_14"], 1),
                "sma_20": _round_or_none(row["sma_20"], 4),
                "sma_50": _round_or_none(row["sma_50"], 4),
                "volatility_pct": _round_or_none(row["volatility_pct"], 2),
                "trend": row["trend"],
            }
        )
    return json.dumps(compact, ensure_ascii=False)

def compact_news_for_prompt(news: list[dict[str, Any]]) -> str:
    """JSON rendering of the scored headlines for the model prompt."""
    compact = [
        {
            "title": item["title"],
            "source": item["source"],
            "sentiment_score": item["sentiment_score"],
            "sentiment": item["sentiment"],
        }
        for item in news
    ]
    return json.dumps(compact, ensure_ascii=False)

def build_analysis_prompt(
    rows: list[dict[str, Any]],
    risk_profile: str,
    max_trade_ideas: int,
    language: str,
    fear_greed: dict[str, Any] | None = None,
    news: list[dict[str, Any]] | None = None,
    news_sentiment: dict[str, Any] | None = None,
    coin_mentions: list[dict[str, Any]] | None = None,
    trading_mode: str = "any",
) -> str:
    """Prompt for the proposal model: data + sentiment + news + instructions.

    ``trading_mode="spot"`` restricts the proposal to what a spot-only trader
    can execute: buy (long) ideas with buy zone, invalidation and sell
    targets, no shorts, no leverage.
    """
    spot_rules = ""
    if trading_mode == "spot":
        spot_rules = (
            "8. HANDELSMODUS: SPOT-ONLY. De handelaar kan uitsluitend kopen en "
            "verkopen op de spotmarkt (geen hefboom, geen shorts, geen "
            "derivaten):\n"
            "   - Sta ALLEEN direction 'long' of 'none' toe. Short-ideeën zijn "
            "niet uitvoerbaar: benoem ze hooguit als context, geef ze nooit als "
            "trade-idee.\n"
            "   - Presenteer de entry-zone als KOOPZONE (limietorders of DCA), "
            "de stop-loss als INVALIDATIENIVEAU (verkooppunt als de prijs er "
            "duidelijk onder zakt) en de take-profits als VERKOOPDOELEN.\n"
            "   - Position sizing als fractie van het spotportfolio, zonder "
            "hefboom; er is geen liquidatierisico, maar een daling kost de "
            "volledige beweging.\n"
        )
    news = news or []
    fg_block = (
        json.dumps(fear_greed, ensure_ascii=False)
        if fear_greed
        else "niet beschikbaar"
    )
    news_block = compact_news_for_prompt(news) if news else "niet beschikbaar"
    news_summary_block = (
        json.dumps(news_sentiment, ensure_ascii=False)
        if news_sentiment
        else "niet beschikbaar"
    )
    mentions_block = (
        json.dumps(coin_mentions, ensure_ascii=False)
        if coin_mentions
        else "geen top-munten in de koppen"
    )
    return (
        "Je bent een nuchtere crypto-marktanalyst. Analyseer uitsluitend de "
        "onderstaande data (CoinGecko/Binance-prijzen, Fear & Greed-sentiment "
        f"en nieuwskoppen van RSS-feeds) en formuleer een trade-voorstel.\n\n"
        f"PRIJSDATA (JSON, candle-interval 4u, top {len(rows)} op market cap):\n"
        f"{compact_rows_for_prompt(rows)}\n\n"
        "MARKT-SENTIMENT - FEAR & GREED INDEX (alternative.me; 0=extreme "
        f"angst, 100=extreme hebzucht; inclusief 1dag/7dag-delta's):\n{fg_block}\n\n"
        f"NIEUWSKOPPEN (laatste ~48u uit RSS-feeds, {len(news)} stuks, met "
        "deterministische keyword-score -2..+2 als ruw signaal):\n"
        f"{news_block}\n\n"
        f"NIEUWS-SAMENVATTING (gemiddelde score en verdeling):\n{news_summary_block}\n\n"
        f"MUNTVERMELDINGEN IN DE KOPPEN:\n{mentions_block}\n\n"
        "REGELS:\n"
        "1. Bepaal eerst het marktbrede regime (bijv. risico-on, risico-off, "
        "rangegebonden) en onderbouw het met prijsdata, sentiment en nieuws.\n"
        f"2. Kies maximaal {max_trade_ideas} van de meest kansrijke trade-setups. "
        "Kwaliteit boven kwantiteit: als er geen overtuigende setup is, geef dan "
        "minder voorstellen of een met direction 'none'.\n"
        "3. Gebruik voor elke setup: direction ('long' of 'short'), "
        "conviction_pct (0-100), een entry-zone (entry_low en entry_high rond de "
        "actuele prijs), een stop-loss, take-profits, een concreet timeframe en "
        "een onderbouwing die verwijst naar de gegevens (momentum, RSI, trend, "
        "volatiliteit, sentiment).\n"
        f"4. Risicoprofiel van de handelaar: {risk_profile}. Stem entry's, "
        "stop-loss-afstanden en position sizing-advies (als fractioneel in de "
        "onderbouwing) daarop af: 'conservative' = kleinere posities en ruimere "
        "stops, 'aggressive' = mag vaker op breakout gokken.\n"
        f"5. Schrijf narrative, regime_rationale, rationale en key_risks in het "
        f"volgende ('nl' = Nederlands, 'en' = Engels): {language}.\n"
        "6. Noem minimaal 2 concrete risico's voor de short-term prijsactie.\n"
        "7. Sentiment en nieuws zijn contextsignalen, geen zekerheden: weeg ze "
        "mee bij regime en onderbouwing, maar speculeer NIET over evenementen, "
        "kortingsacties of kennis die niet in de aangeleverde data staan.\n"
        + spot_rules
    )

def _fmt_price(value: float | int | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value == 0:
        return "0"
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    decimals = max(2, min(8, 2 - int(math.floor(math.log10(abs(value))))))
    return f"{value:.{decimals}f}"

def _fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.2f}%"

def _fmt_num(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"

def _fmt_signed(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:+d}"

def _fmt_news_date(iso: str | None) -> str:
    """Short 'dd-mm hh:mm' rendering of a news item's ISO timestamp."""
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso).strftime("%d-%m %H:%M")
    except ValueError:
        return "-"

def render_report(
    as_of: str,
    quote_currency: str,
    risk_profile: str,
    rows: list[dict[str, Any]],
    proposal: dict[str, Any],
    model: str,
    fear_greed: dict[str, Any] | None = None,
    news: list[dict[str, Any]] | None = None,
    news_sentiment: dict[str, Any] | None = None,
    coin_mentions: list[dict[str, Any]] | None = None,
    trading_mode: str = "any",
) -> str:
    """Render the final Markdown report from snapshot, proposal and sentiment.

    In ``trading_mode="spot"`` the field labels shift to spot terminology and
    short ideas (which the prompt already forbids) render as not executable.
    """
    regime = proposal.get("market_regime", "onbekend")
    news = news or []
    has_sentiment = fear_greed is not None or bool(news)
    spot = trading_mode == "spot"

    data_sources = "CoinGecko + Binance"
    if fear_greed is not None:
        data_sources += " + F&G (alternative.me)"
    if news:
        data_sources += " + RSS-nieuws (Cointelegraph/Decrypt/CoinDesk)"

    mode_line = f"Risicoprofiel: {risk_profile}"
    if spot:
        mode_line += " · Handelsmodus: spot (alleen koop, geen hefboom)"

    lines: list[str] = [
        "# Crypto Marktscan - Trade-voorstel",
        "",
        f"*Van: {as_of} ·Quote: {quote_currency.upper()} · "
        f"{mode_line} · Model: {model} · "
        f"Data: {data_sources}*",
        "",
        "## Samenvatting",
        "",
        str(proposal.get("narrative", "")),
        "",
        "## Marktbreed regime",
        "",
        f"**{regime}** - {proposal.get('regime_rationale', '')}",
        "",
    ]

    if has_sentiment:
        lines.extend(["## Sentiment", ""])
        if fear_greed:
            fg_value = fear_greed.get("value")
            fg_line = (
                f"**Fear & Greed Index:** {fg_value}/100 "
                f"({fear_greed.get('classification', 'onbekend')})"
            )
            if fear_greed.get("value_yesterday") is not None:
                fg_line += (
                    f" · 24u eerder: {fear_greed['value_yesterday']} "
                    f"({_fmt_signed(fear_greed.get('change_1d'))})"
                )
            if fear_greed.get("value_last_week") is not None:
                fg_line += (
                    f" · week eerder: {fear_greed['value_last_week']} "
                    f"({_fmt_signed(fear_greed.get('change_7d'))})"
                )
            lines.extend([fg_line, ""])
        else:
            lines.extend(["Fear & Greed Index was deze scan niet beschikbaar.", ""])

        if news_sentiment:
            lines.append(
                f"**Nieuwssentiment:** {news_sentiment['label']} "
                f"(gem. score {news_sentiment['average_score']:+.2f}; "
                f"{news_sentiment['positive']} positief, "
                f"{news_sentiment['negative']} negatief, "
                f"{news_sentiment['neutral']} neutraal van "
                f"{news_sentiment['total']} koppen)"
            )
            lines.append("")

        if news:
            if coin_mentions:
                mentions = ", ".join(
                    f"{m['symbol']} ({m['count']}x)" for m in coin_mentions
                )
                lines.extend([f"**Meest genoemd in het nieuws:** {mentions}", ""])
            else:
                lines.extend(["Geen van de top-munten staat in de koppen.", ""])

        lines.extend(
            [
                "## Nieuws (laatste ~48 uur)",
                "",
                "| Kop | Bron | Tijd | Sentiment |",
                "|-----|------|------|-----------|",
            ]
        )
        for item in news:
            title = item["title"].replace("|", "/")
            lines.append(
                f"| {title} | {item['source']} | {_fmt_news_date(item.get('published'))} "
                f"| {item['sentiment']} ({_fmt_signed(item['sentiment_score'])}) |"
            )
        lines.append("")

    ideas = proposal.get("trade_ideas") or []
    if not ideas:
        lines.append("Geen voorstellen deze scan.")
    if spot:
        lines.extend(
            [
                "*Handelsmodus spot: alleen koopideeën (geen hefboom, geen shorts). "
                "Koopzone = limietorders of DCA; invalidatie = verkooppunt als de "
                "prijs eronder zakt; verkoopdoelen = deelverkopen bij de targets.*",
                "",
            ]
        )
    entry_label = "Koopzone" if spot else "Entry-zone"
    stop_label = "Invalidatie (verkoop onder)" if spot else "Stop-loss"
    tp_label = "Verkoopdoelen" if spot else "Take-profit"
    for idx, idea in enumerate(ideas, start=1):
        direction = str(idea.get("direction", "none")).upper()
        conviction = idea.get("conviction_pct", 0)
        lines.append(f"### {idx}. {idea.get('symbol', '?')} - {direction} "
                      f"(overtuiging {conviction}/100)")
        lines.append("")
        if direction == "NONE" or (spot and direction == "SHORT"):
            if spot and direction == "SHORT":
                lines.append(
                    f"Geen setup (short is op de spotmarkt niet uitvoerbaar): "
                    f"{idea.get('rationale', '')}"
                )
            else:
                lines.append(f"Geen setup: {idea.get('rationale', '')}")
        else:
            lines.append(f"- **{entry_label}:** {_fmt_price(idea.get('entry_low'))} "
                         f"- {_fmt_price(idea.get('entry_high'))}")
            lines.append(f"- **{stop_label}:** {_fmt_price(idea.get('stop_loss'))}")
            tp1 = _fmt_price(idea.get("take_profit_1"))
            tp2 = _fmt_price(idea.get("take_profit_2"))
            lines.append(f"- **{tp_label}:** TP1 {tp1}"
                         + (f" · TP2 {tp2}" if idea.get("take_profit_2") is not None else ""))
            lines.append(f"- **Timeframe:** {idea.get('timeframe', '-')}")
            lines.append(f"- **Onderbouwing:** {idea.get('rationale', '-')}")
        lines.append("")

    lines.extend(
        [
            f"## Top {len(rows)} op market cap",
            "",
            "| # | Coin | Prijs | 24u | 7d | 30d | RSI | Trend |",
            "|---|------|-------|-----|-----|-----|-----|-------|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['rank'] or '-'} | {row['name']} ({row['symbol']}) "
            f"| {_fmt_price(row['price'])} "
            f"| {_fmt_pct(row['change_24h_pct'])} "
            f"| {_fmt_pct(row['change_7d_pct'])} "
            f"| {_fmt_pct(row['change_30d_pct'])} "
            f"| {_fmt_num(row['rsi_14'])} "
            f"| {row['trend']} |"
        )

    lines.extend(["", "## Belangrijkste risico's", ""])
    for risk in proposal.get("key_risks") or []:
        lines.append(f"- {risk}")
    if not proposal.get("key_risks"):
        lines.append("- (geen risico's geretourneerd)")

    lines.extend(
        [
            "",
            "> **Disclaimer:** dit is een geautomatiseerde analyse op basis van "
            "publieke marktdata, sentiment en nieuwskoppen, geen financieel "
            "advies. Crypto is volatiel; verlies van je inleg is mogelijk. "
            "Doe altijd eigen onderzoek.",
        ]
    )
    return "\n".join(lines)
