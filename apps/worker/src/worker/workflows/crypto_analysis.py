"""Pure analysis helpers for the crypto market scan workflow.

This module deliberately depends only on the Python standard library: the
functions are deterministic and side-effect free, so they can run inside the
workflow body (the determinism sandbox) and be unit-tested without the
Workflows SDK installed. All network I/O lives in ``crypto_scan.py``.
"""

from __future__ import annotations

import json
import math
from typing import Any

# Top fiat-pegged coins: trading them against USDT is pointless and their
# TA values are noise, so the workflow filters them out before analysis.
STABLE_SYMBOLS = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USDS"}

MAX_TOKENS_ROWS = 30  # rows kept for the model prompt, hard safety bound


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


def build_analysis_prompt(
    rows: list[dict[str, Any]],
    risk_profile: str,
    max_trade_ideas: int,
    language: str,
) -> str:
    """Prompt for the proposal model: data + role + instructions + language."""
    return (
        "Je bent een nuchtere crypto-marktanalist. Analyseer uitsluitend de "
        "onderstaande data uit de CoinGecko- en Binance-API's (top coins op "
        f"market cap, {len(rows)} stuks) en formuleer een trade-voorstel.\n\n"
        f"DATA (JSON, candle-interval 4u):\n{compact_rows_for_prompt(rows)}\n\n"
        "REGELS:\n"
        "1. Bepaal eerst het marktbrede regime (bijv. risico-on, risico-off, "
        "rangegebonden) en onderbouw het met de data.\n"
        f"2. Kies maximaal {max_trade_ideas} van de meest kansrijke trade-setups. "
        "Kwaliteit boven kwantiteit: als er geen overtuigende setup is, geef dan "
        "minder voorstellen of een met direction 'none'.\n"
        "3. Gebruik voor elke setup: direction ('long' of 'short'), "
        "conviction_pct (0-100), een entry-zone (entry_low en entry_high rond de "
        "actuele prijs), een stop-loss, take-profits, een concreet timeframe en "
        "een onderbouwing die verwijst naar de gegevens (momentum, RSI, trend, "
        "volatiliteit).\n"
        f"4. Risicoprofiel van de handelaar: {risk_profile}. Stem entry's, "
        "stop-loss-afstanden en position sizing-advies (als fractioneel in de "
        "onderbouwing) daarop af: 'conservative' = kleinere posities en ruimere "
        "stops, 'aggressive' = mag vaker op breakout gokken.\n"
        f"5. Schrijf narrative, regime_rationale, rationale en key_risks in het "
        f"volgende ('nl' = Nederlands, 'en' = Engels): {language}.\n"
        "6. Noem minimaal 2 concrete risico's voor de short-term prijsactie.\n"
        "7. Baseer je NIET op kennis die niet in de data staat; speculeer niet "
        "over nieuws of evenementen.\n"
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


def render_report(
    as_of: str,
    quote_currency: str,
    risk_profile: str,
    rows: list[dict[str, Any]],
    proposal: dict[str, Any],
    model: str,
) -> str:
    """Render the final Markdown report from the snapshot and the proposal."""
    regime = proposal.get("market_regime", "onbekend")
    lines: list[str] = [
        "# Crypto Marktscan - Trade-voorstel",
        "",
        f"*Van: {as_of} ·Quote: {quote_currency.upper()} · "
        f"Risicoprofiel: {risk_profile} · Model: {model} · "
        "Data: CoinGecko + Binance*",
        "",
        "## Samenvatting",
        "",
        str(proposal.get("narrative", "")),
        "",
        "## Marktbreed regime",
        "",
        f"**{regime}** - {proposal.get('regime_rationale', '')}",
        "",
        "## Trade-voorstellen",
        "",
    ]

    ideas = proposal.get("trade_ideas") or []
    if not ideas:
        lines.append("Geen voorstellen deze scan.")
    for idx, idea in enumerate(ideas, start=1):
        direction = str(idea.get("direction", "none")).upper()
        conviction = idea.get("conviction_pct", 0)
        lines.append(f"### {idx}. {idea.get('symbol', '?')} - {direction} "
                      f"(overtuiging {conviction}/100)")
        lines.append("")
        if direction == "NONE":
            lines.append(f"Geen setup: {idea.get('rationale', '')}")
        else:
            lines.append(f"- **Entry-zone:** {_fmt_price(idea.get('entry_low'))} "
                         f"- {_fmt_price(idea.get('entry_high'))}")
            lines.append(f"- **Stop-loss:** {_fmt_price(idea.get('stop_loss'))}")
            tp1 = _fmt_price(idea.get("take_profit_1"))
            tp2 = _fmt_price(idea.get("take_profit_2"))
            lines.append(f"- **Take-profit:** TP1 {tp1}"
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
            "publieke marktdata, geen financieel advies. Crypto is volatiel; "
            "verlies van je inleg is mogelijk. Doe altijd eigen onderzoek.",
        ]
    )
    return "\n".join(lines)
