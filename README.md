# Crypto Market Agent

Een Mistral Workflow die om de 6 uur de cryptomarkt scant en een onderbouwd
trade-voorstel genereert. De workflow draait als managed deployment op het
Mistral-platform en produceert een rapport; hij plaatst nooit orders.

## Wat hij doet

1. **Top coins ophalen** - CoinGecko `/coins/markets`: de top N (standaard 20)
   op market cap, inclusief 24u/7d/30d prijsveranderingen. Stablecoins worden
   overgeslagen.
2. **Technische analyse** - Binance `/api/v3/klines` (gratis publiek, geen API-key):
   4u-candles waaruit RSI(14), SMA20, SMA50, volatiliteit en een trendlabel
   worden berekend. Mist een coin een USDT-paar, dan krijgt die alleen de
   CoinGecko-momentumdata.
3. **Trade-voorstel** - Een Mistral-model (standaard `mistral-medium-latest`)
   zet de metrics om in een gestructureerd voorstel: marktbreed regime, tot 3
   trade-ideeën met entry-zone, stop-loss, take-profits, timeframe en
   onderbouwing, plus de belangrijkste risico's.
4. **Rapport** - De workflow rendert een Markdown-rapport (`report_markdown`)
   plus de gestructureerde data als uitvoer van de execution.

## Workflow

Registratienaam: `crypto-market-scan`

```text
Input (ScanInput):  top_n, quote_currency, klines_interval, klines_limit,
                    model, risk_profile (conservative|balanced|aggressive),
                    report_language, max_trade_ideas
Output (ScanOutput): as_of, market_snapshot[], proposal, report_markdown
```

- Code: `apps/worker/src/worker/workflows/crypto_scan.py`
- Pure analysefuncties (unit-testbaar, stdlib-only):
  `apps/worker/src/worker/workflows/crypto_analysis.py`
- Tests: `apps/worker/tests/test_crypto_analysis.py`
  (`python3 apps/worker/tests/test_crypto_analysis.py`)

## Lokaal draaien

```bash
# .env invullen met MISTRAL_API_KEY (zie .env.example)
cd apps/worker
uv run --all-packages python ../../.agents/skills/capability-workflows/scripts/test_workflow.py \
  src/worker/workflows/crypto_scan.py --input '{"top_n": 20}' --timeout 60
```

## Deployment

Managed deployment vanaf deze repo (build context: root, Dockerfile:
`deploy/docker/Dockerfile.worker`). Het schema (elke 6 uur) wordt via de
Workflows-API op de deployment gezet, niet in de worker.

## Data & kosten

- CoinGecko en Binance publieke API's: gratis, geen credentials nodig.
- Per run: 1 CoinGecko-call, ~19 Binance-calls, 1 modelcall (medium).

## Disclaimer

Geautomatiseerde analyse op publieke marktdata, geen financieel advies.
Crypto is volatiel; verlies van je inleg is mogelijk.
