# AI + Blockchain Integration & Smart Automation (Windows + VS Code Demo) — No PowerShell, No pip

This folder is designed for **locked-down student machines**:
- No `.ps1` scripts
- No third‑party Python packages
- Requires only **Python 3.8+** (3.10+ recommended)

It **reads** live blockchain data (it does **not** send transactions and uses **no private keys**).

## What it does
- Connects to Ethereum **Sepolia** via JSON‑RPC (with failover + retry)
- Pulls blocks + transactions, and (when relevant) receipts
- Enriches:
  - contract creation vs. interaction
  - gas used + success/failure
  - ERC‑20 `Transfer` events (from/to/token + raw value)
  - local watchlist + labels
- Scores risk with explainable reasons
- Produces an **AI‑style triage summary**:
  - Default: heuristic “analyst note” (always available)
  - Optional: call your org’s LLM HTTP endpoint (if provided)
- Outputs:
  - `alerts.jsonl` (one JSON per line — SIEM/log shipper friendly)
  - Optional webhook POST per alert
  - `state.db` SQLite state to avoid reprocessing the same blocks in `--follow` mode

## Quick start (students)
1) Open this folder in **VS Code**
2) VS Code Terminal (Command Prompt or PowerShell) run:

```bat
py ai_chain_bot.py --once
```

Alternative one‑click:

```bat
run_demo.cmd
```

## Modes
### One‑off scan (last N blocks)
```bat
py ai_chain_bot.py --once
```

### Follow mode (polls for new blocks)
```bat
py ai_chain_bot.py --follow
```

## The only configuration file (no code edits)
Edit `config.json`:
- `lookback_blocks` (how far back `--once` scans)
- `min_score` (alert threshold)
- `value_threshold_eth` (“high value” cutoff)
- `watchlist_path` (addresses to boost score)
- `webhook_url` (optional)
- `rpc_urls` (add/reorder for reliability)
- `ai.summary_mode` = `"heuristic"` (default) or `"llm"` (optional)

## Files you’ll see after running
- `alerts.jsonl` — alert output (append‑only)
- `state.db` — SQLite state (safe to delete if you want a clean run)

## Troubleshooting
- If `py` isn’t available, use: `python ai_chain_bot.py --once`
- If an RPC is slow, reorder/add URLs in `config.json`


## Note
If you ever see a ValueError converting `"0x"` to int, update to the latest ZIP — this demo treats `"0x"` as zero.
