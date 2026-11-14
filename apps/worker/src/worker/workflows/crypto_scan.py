"""Crypto market scan: scans the top coins on market cap and produces a trade proposal.

Runs unattended on a schedule (every 6 hours) as a managed deployment. It uses
only free public data: CoinGecko for the top-N ranking and price changes,
Binance for OHLCV candles, alternative.me for the Fear & Greed Index and
public RSS feeds (Cointelegraph, Decrypt, CoinDesk) for news headlines. A
Mistral model turns prices, sentiment and news into a structured, reasoned
trade proposal. The workflow never places orders; its output is advisory.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    chat_parse_to_model,
)

from worker.workflows import crypto_analysis

RISK_PROFILES = ("conservative", "balanced", "aggressive")
TRADING_MODES = ("any", "spot")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ScanInput(BaseModel):
    """Input for one market scan."""

    top_n: int = Field(default=20, ge=5, le=50)
    quote_currency: str = "usd"
    klines_interval: str = "4h"
    klines_limit: int = Field(default=180, ge=60, le=1000)
    model: str = "mistral-medium-latest"
    risk_profile: str = "balanced"
    report_language: str = "nl"
    max_trade_ideas: int = Field(default=3, ge=1, le=5)
    news_max_items: int = Field(default=12, ge=5, le=25)
    trading_mode: str = Field(
        default="any",
        description=(
            "'any' = long/short zoals de data dicteert; 'spot' = alleen koop-"
            "ideeen (buy zone, invalidatie, verkoopdoelen), shorts niet uitvoerbaar"
        ),
    )


class CoinSnapshot(BaseModel):
    """Per-coin metrics as used in the report and prompt."""

    symbol: str
    name: str
    rank: int
    price: float
    market_cap: float
    volume_24h: float
    change_24h_pct: float | None = None
    change_7d_pct: float | None = None
    change_30d_pct: float | None = None
    rsi_14: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    volatility_pct: float | None = None
    trend: str = "neutral"
    klines_source: str = "none"


class FearGreed(BaseModel):
    """Market-wide sentiment state from alternative.me."""

    value: int = Field(ge=0, le=100)
    classification: str
    value_yesterday: int | None = None
    value_last_week: int | None = None
    change_1d: int | None = None
    change_7d: int | None = None
    history: list[int] = Field(default_factory=list)


class NewsItem(BaseModel):
    """One scored news headline from the public RSS feeds."""

    title: str
    source: str
    published: str | None = None
    url: str = ""
    sentiment_score: int = Field(ge=-2, le=2)
    sentiment: str


class TradeIdea(BaseModel):
    """One concrete trade setup proposed by the model."""

    symbol: str
    direction: str = Field(description="'long', 'short' or 'none'")
    conviction_pct: int = Field(ge=0, le=100)
    entry_low: float | None = None
    entry_high: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    timeframe: str = "-"
    rationale: str = ""


class Proposal(BaseModel):
    """Structured trade proposal returned by the model."""

    market_regime: str
    regime_rationale: str
    narrative: str
    trade_ideas: list[TradeIdea]
    key_risks: list[str]


class ScanOutput(BaseModel):
    """Final scan result: structured data plus a Markdown report."""

    as_of: str
    quote_currency: str
    risk_profile: str
    trading_mode: str = "any"
    model: str
    market_snapshot: list[CoinSnapshot]
    fear_greed: FearGreed | None = None
    news: list[NewsItem] = Field(default_factory=list)
    news_sentiment: dict | None = None
    proposal: Proposal
    report_markdown: str


@workflows.activity()
async def fetch_top_markets(top_n: int, quote_currency: str) -> list[dict]:
    """Fetch the top coins by market cap from CoinGecko (free public API)."""
    import httpx

    params = {
        "vs_currency": quote_currency,
        "order": "market_cap_desc",
        "per_page": top_n,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h,7d,30d",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        last_error = "no attempt made"
        for attempt in range(4):
            resp = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets", params=params
            )
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}"
            if resp.status_code in _RETRYABLE_STATUS:
                await asyncio.sleep(1.5 * (2**attempt))
                continue
            resp.raise_for_status()
        raise RuntimeError(f"CoinGecko markets failed after retries: {last_error}")


@workflows.activity()
async def fetch_klines(symbols: list[str], interval: str, limit: int) -> dict[str, dict]:
    """Fetch USDT klines from Binance (free public API); skip missing pairs.

    Returns ``{symbol: {closes: [...], volumes: [...]}}`` for the symbols that
    have a liquid pair; a coin without a result simply gets no TA values.
    """
    import httpx

    result: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for symbol in symbols:
            for attempt in range(3):
                try:
                    resp = await client.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": symbol, "interval": interval, "limit": limit},
                    )
                except httpx.HTTPError:
                    break  # network hiccup: no TA for this coin, not fatal
                if resp.status_code == 200:
                    parsed = crypto_analysis.parse_binance_klines(resp.json())
                    if parsed["closes"]:
                        result[symbol] = parsed
                    break
                if resp.status_code in _RETRYABLE_STATUS:
                    await asyncio.sleep(1.5 * (2**attempt))
                    continue
                break  # 400/404: pair does not exist, skip the coin
    return result


@workflows.activity()
async def fetch_fear_greed() -> dict | None:
    """Fetch the crypto Fear & Greed Index (alternative.me, free public API).

    Returns ``None`` when the service is unreachable: sentiment enriches the
    analysis, but a scan without it is still valid, so failure is not fatal.
    """
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                resp = await client.get(
                    "https://api.alternative.me/fng/", params={"limit": 8}
                )
            except httpx.HTTPError:
                break
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in _RETRYABLE_STATUS:
                await asyncio.sleep(1.5 * (2**attempt))
                continue
            break
    return None


@workflows.activity()
async def fetch_news_feeds() -> dict[str, str]:
    """Fetch the raw RSS XML of each public crypto news feed.

    Returns ``{source: xml_text}`` for the feeds that responded; an
    unreachable feed is simply absent and contributes no headlines.
    """
    import httpx

    feeds: dict[str, str] = {}
    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": "crypto-market-agent/1.0 (+rss)"}
    ) as client:
        for source, url in crypto_analysis.NEWS_FEEDS.items():
            try:
                resp = await client.get(url)
            except httpx.HTTPError:
                continue  # skip this feed, the others may still work
            if resp.status_code == 200 and resp.text:
                feeds[source] = resp.text
    return feeds


@workflows.workflow.define(
    name="crypto-market-scan",
    workflow_display_name="Crypto Market Scan",
    workflow_description=(
        "Scant de top crypto's op market cap (CoinGecko + Binance data), "
        "verrijkt de analyse met markt-sentiment (Fear & Greed Index) en "
        "cryptonieuws uit RSS-feeds, en produceert met een Mistral-model een "
        "onderbouwd trade-voorstel met entry, stop-loss en take-profits. "
        "Ondersteunt een spot-modus (trading_mode='spot') voor handelaren die "
        "alleen spot kunnen traden: dan alleen koop-ideeën met koopzone, "
        "invalidatieniveau en verkoopdoelen. Advies, geen orders."
    ),
)
class CryptoMarketScan:
    @workflows.workflow.entrypoint
    async def run(self, input: ScanInput) -> ScanOutput:
        if input.risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"risk_profile moet een van {RISK_PROFILES} zijn, got: {input.risk_profile}"
            )
        if input.trading_mode not in TRADING_MODES:
            raise ValueError(
                f"trading_mode moet een van {TRADING_MODES} zijn, got: {input.trading_mode}"
            )

        markets = await fetch_top_markets(input.top_n, input.quote_currency)
        coins = [
            m
            for m in markets
            if str(m.get("symbol", "")).upper() not in crypto_analysis.STABLE_SYMBOLS
        ]
        as_of = crypto_analysis.latest_timestamp(coins)
        if not coins:
            raise RuntimeError("Geen bruikbare coins gevonden in de CoinGecko-response.")

        symbols = [f"{str(c.get('symbol', '')).upper()}USDT" for c in coins]
        klines = await fetch_klines(symbols, input.klines_interval, input.klines_limit)

        rows = [
            crypto_analysis.build_snapshot_row(coin, klines.get(symbol))
            for coin, symbol in zip(coins, symbols, strict=False)
        ]

        fear_greed_raw = await fetch_fear_greed()
        fear_greed = (
            crypto_analysis.parse_fear_greed(fear_greed_raw) if fear_greed_raw else None
        )

        feeds_xml = await fetch_news_feeds()
        news = crypto_analysis.merge_news(
            {
                source: crypto_analysis.parse_rss_feed(xml_text, source)
                for source, xml_text in feeds_xml.items()
            },
            max_items=input.news_max_items,
        )
        news_sentiment = crypto_analysis.summarize_news_sentiment(news)
        coin_mentions = crypto_analysis.coin_mention_counts(news, rows)

        prompt = crypto_analysis.build_analysis_prompt(
            rows,
            input.risk_profile,
            input.max_trade_ideas,
            input.report_language,
            fear_greed=fear_greed,
            news=news,
            news_sentiment=news_sentiment,
            coin_mentions=coin_mentions,
            trading_mode=input.trading_mode,
        )
        request = ChatCompletionRequest(
            model=input.model,
            messages=[UserMessage(content=prompt)],
        )
        proposal = await chat_parse_to_model(Proposal, request)

        report_markdown = crypto_analysis.render_report(
            as_of=as_of,
            quote_currency=input.quote_currency,
            risk_profile=input.risk_profile,
            rows=rows,
            proposal=proposal.model_dump(),
            model=input.model,
            fear_greed=fear_greed,
            news=news,
            news_sentiment=news_sentiment,
            coin_mentions=coin_mentions,
            trading_mode=input.trading_mode,
        )

        return ScanOutput(
            as_of=as_of,
            quote_currency=input.quote_currency,
            risk_profile=input.risk_profile,
            trading_mode=input.trading_mode,
            model=input.model,
            market_snapshot=[CoinSnapshot(**row) for row in rows],
            fear_greed=(
                FearGreed(**fear_greed) if fear_greed is not None else None
            ),
            news=[NewsItem(**item) for item in news],
            news_sentiment=news_sentiment,
            proposal=proposal,
            report_markdown=report_markdown,
        )
