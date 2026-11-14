"""Crypto market scan: scans the top coins on market cap and produces a trade proposal.

Runs unattended on a schedule (every 6 hours) as a managed deployment. It uses
only free public market data: CoinGecko for the top-N ranking and price
changes, Binance for OHLCV candles, and a Mistral model to turn the numbers
into a structured, reasoned trade proposal. The workflow never places orders;
its output is an advisory report only.
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
    model: str
    market_snapshot: list[CoinSnapshot]
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


@workflows.workflow.define(
    name="crypto-market-scan",
    workflow_display_name="Crypto Market Scan",
    workflow_description=(
        "Scant de top crypto's op market cap (CoinGecko + Binance data) en "
        "produceert met een Mistral-model een onderbouwd trade-voorstel met "
        "entry, stop-loss en take-profits. Advies, geen orders."
    ),
)
class CryptoMarketScan:
    @workflows.workflow.entrypoint
    async def run(self, input: ScanInput) -> ScanOutput:
        if input.risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"risk_profile moet een van {RISK_PROFILES} zijn, got: {input.risk_profile}"
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

        prompt = crypto_analysis.build_analysis_prompt(
            rows, input.risk_profile, input.max_trade_ideas, input.report_language
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
        )

        return ScanOutput(
            as_of=as_of,
            quote_currency=input.quote_currency,
            risk_profile=input.risk_profile,
            model=input.model,
            market_snapshot=[CoinSnapshot(**row) for row in rows],
            proposal=proposal,
            report_markdown=report_markdown,
        )
