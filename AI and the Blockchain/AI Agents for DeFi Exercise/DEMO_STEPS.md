# Step-by-step demo guide (Windows + VS Code)

## 0) What you’ll say (15 seconds)
“We’re going to watch a live blockchain (Sepolia), enrich raw transactions with receipts + token events, score them with explainable rules, and output automation-ready alerts.”

---

## 1) Open + run (2 minutes)
1. VS Code → **File → Open Folder…** → select this folder
2. Terminal (bottom) run:

```bat
py ai_chain_bot.py --once
```

What the audience sees:
- A few `[ALERT] ...` lines in the console
- New file: `alerts.jsonl`

---

## 2) Explain what the script is doing (2–3 minutes)
Use this mental model:

**Ingest**
- `eth_blockNumber` gets the latest block
- `eth_getBlockByNumber(..., true)` pulls transactions

**Enrich**
- “Interesting” transactions also get receipts via `eth_getTransactionReceipt`
- Receipt adds: `status`, `gasUsed`, and `logs`
- Logs are scanned for ERC‑20 `Transfer` events

**AI triage**
- Rule-based risk score + reasons
- An “AI-style” summary is generated (heuristic by default)
- Optional LLM mode (if your org provides an endpoint)

**Automation output**
- Writes JSON lines to `alerts.jsonl`
- Optional webhook POST per alert

---

## 3) Show the output file (1 minute)
1. Open `alerts.jsonl` in VS Code.
2. Point out fields:
   - `score`, `reasons`
   - `value_eth`, `gas_used`, `method_selector`
   - `erc20_transfers` (when present)
   - `ai_summary`

---

## 4) Entry-level exercise: tune the threshold (4–5 minutes)
1. Open `config.json`
2. Change:
   - `min_score`: 60 → 45
3. Run again:

```bat
py ai_chain_bot.py --once
```

Expected:
- More alerts appear because the filter is less strict.

Teaching point:
- “Automation begins with sane thresholds; too low = noise, too high = misses.”

---

## 5) Entry-level exercise: watchlist boost (4–5 minutes)
1. Copy a `from` or `to` address from an alert
2. Paste it into `watchlist.txt` (one address per line)
3. Run again:

```bat
py ai_chain_bot.py --once
```

Expected:
- Alerts involving that address jump in score
- A new reason appears: `Watchlist match`

Teaching point:
- “This is how investigations teams plug intelligence into automation.”

---

## 6) Advanced: follow mode (3–5 minutes)
Run:

```bat
py ai_chain_bot.py --follow
```

Expected:
- Every few seconds you’ll see “new blocks …” and occasional alerts.

Teaching point:
- “This is the real-time backbone for smart automation.”

Stop it with **Ctrl+C**.

---

## 7) Advanced: add a label (2 minutes)
1. Edit `labels.csv`:
   - Add `address,label` entry for an address you’ve seen
2. Re-run `--once`

Expected:
- Console shows friendlier `from_label/to_label` in alerts.

Teaching point:
- “Labels are lightweight enrichment that makes triage faster.”

---

## 8) Optional automation: webhook (if you have one) (2 minutes)
In `config.json`, set:

```json
"webhook_url": "https://YOUR_WEBHOOK_ENDPOINT"
```

Run `--once`. The script POSTs each alert as JSON.

Teaching point:
- “This is how you connect to Teams/Slack, SOAR, ticketing, etc.”

---

## 9) Optional AI: LLM summary mode (if your org provides it) (2 minutes)
In `config.json`:

- `ai.summary_mode` → `"llm"`
- set `ai.llm_endpoint` and `ai.llm_api_key`

If not configured, it automatically falls back to heuristic summary.

---

## 10) Reset for a clean rerun (10 seconds)
Delete:
- `alerts.jsonl`
- `state.db`

Then run `--once` again.
