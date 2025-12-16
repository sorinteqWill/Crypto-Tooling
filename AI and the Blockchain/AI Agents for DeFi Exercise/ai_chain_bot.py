import argparse
import csv
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib import request


# ERC-20 Transfer(address,address,uint256) topic0 (keccak hash)
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def utc_iso(ts: Optional[int] = None) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def hex_to_int(x) -> int:
    """
    Safely convert JSON-RPC hex strings (e.g., "0x1a") to int.

    Some nodes occasionally return "0x" for zero; treat that as 0.
    Also tolerates None and decimal strings.
    """
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("", "0x"):
            return 0
        if s.startswith("0x"):
            return int(s, 16)
        return int(s)
    return int(x)



def wei_to_eth(wei: int) -> float:
    return wei / 10**18


def norm_addr(a: Optional[str]) -> Optional[str]:
    if a is None:
        return None
    a = a.strip()
    return a.lower() if a.startswith("0x") else a


def topic_to_addr(topic_hex: str) -> str:
    # topics are 32 bytes; address is last 20 bytes
    return "0x" + topic_hex[-40:]


@dataclass
class Config:
    chain: str
    rpc_urls: List[str]
    lookback_blocks: int
    follow_poll_seconds: int
    min_score: int
    value_threshold_eth: float
    high_gas_threshold: int
    max_receipt_fetch_per_block: int
    output_jsonl: str
    sqlite_state_db: str
    watchlist_path: str
    labels_path: str
    webhook_url: str
    ai_enable_summary: bool
    ai_summary_mode: str
    llm_endpoint: str
    llm_api_key: str
    llm_timeout_seconds: int


def load_config(path: str = "config.json") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ai = raw.get("ai", {}) or {}
    return Config(
        chain=raw.get("chain", "sepolia"),
        rpc_urls=list(raw.get("rpc_urls", [])),
        lookback_blocks=int(raw.get("lookback_blocks", 20)),
        follow_poll_seconds=int(raw.get("follow_poll_seconds", 8)),
        min_score=int(raw.get("min_score", 60)),
        value_threshold_eth=float(raw.get("value_threshold_eth", 0.05)),
        high_gas_threshold=int(raw.get("high_gas_threshold", 300000)),
        max_receipt_fetch_per_block=int(raw.get("max_receipt_fetch_per_block", 9999)),
        output_jsonl=str(raw.get("output_jsonl", "alerts.jsonl")),
        sqlite_state_db=str(raw.get("sqlite_state_db", "state.db")),
        watchlist_path=str(raw.get("watchlist_path", "watchlist.txt")),
        labels_path=str(raw.get("labels_path", "labels.csv")),
        webhook_url=str(raw.get("webhook_url", "")),
        ai_enable_summary=bool(ai.get("enable_summary", True)),
        ai_summary_mode=str(ai.get("summary_mode", "heuristic")),
        llm_endpoint=str(ai.get("llm_endpoint", "")),
        llm_api_key=str(ai.get("llm_api_key", "")),
        llm_timeout_seconds=int(ai.get("llm_timeout_seconds", 30)),
    )


def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = request.Request(url, data=data, headers=req_headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


class RpcClient:
    def __init__(self, rpc_urls: List[str], timeout: int = 30):
        if not rpc_urls:
            raise ValueError("No rpc_urls configured.")
        self.rpc_urls = rpc_urls
        self.timeout = timeout
        self._idx = 0

    def _current(self) -> str:
        return self.rpc_urls[self._idx % len(self.rpc_urls)]

    def _rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self.rpc_urls)

    def call(self, method: str, params: List[Any]) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(5):
            url = self._current()
            try:
                payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                data = http_post_json(url, payload, timeout=self.timeout)
                if "error" in data:
                    raise RuntimeError(data["error"])
                return data["result"]
            except Exception as e:
                last_err = e
                self._rotate()
                time.sleep(0.5 + attempt * 0.5)
        raise RuntimeError(f"RPC failed for {method} after retries: {last_err}")

    def latest_block(self) -> int:
        return hex_to_int(self.call("eth_blockNumber", []))

    def get_block_with_txs(self, bn: int) -> Dict[str, Any]:
        return self.call("eth_getBlockByNumber", [hex(bn), True])

    def get_receipt(self, tx_hash: str) -> Dict[str, Any]:
        return self.call("eth_getTransactionReceipt", [tx_hash])


def load_watchlist(path: str) -> set:
    wl = set()
    if not os.path.exists(path):
        return wl
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            wl.add(norm_addr(s))
    return wl


def load_labels(path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if row.strip() and not row.startswith("#"))
        for row in reader:
            a = norm_addr(row.get("address", ""))
            if a and a.startswith("0x") and row.get("label"):
                labels[a] = row["label"].strip()
    return labels


def ensure_state_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT)")
    conn.commit()
    return conn


def get_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    cur = conn.execute("SELECT v FROM state WHERE k=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def set_state(conn: sqlite3.Connection, key: str, val: str) -> None:
    conn.execute(
        "INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, val),
    )
    conn.commit()


def parse_erc20_transfers(receipt: Dict[str, Any]) -> List[Dict[str, Any]]:
    transfers: List[Dict[str, Any]] = []
    for log in receipt.get("logs", []) or []:
        topics = [t.lower() for t in (log.get("topics") or [])]
        if not topics or topics[0] != TRANSFER_TOPIC0:
            continue
        if len(topics) < 3:
            continue
        token = norm_addr(log.get("address"))
        frm = norm_addr(topic_to_addr(topics[1]))
        to = norm_addr(topic_to_addr(topics[2]))
        value_raw = hex_to_int(log.get("data", "0x0"))
        transfers.append({"token": token, "from": frm, "to": to, "value_raw": value_raw})
    return transfers


def classify_method(selector: Optional[str]) -> Optional[str]:
    if not selector:
        return None
    known = {
        "0x095ea7b3": "ERC-20 approve",
        "0xa9059cbb": "ERC-20 transfer",
        "0x23b872dd": "ERC-20 transferFrom",
    }
    return known.get(selector.lower())


def score_transaction(
    tx: Dict[str, Any],
    receipt: Optional[Dict[str, Any]],
    cfg: Config,
    watchlist: set,
    labels: Dict[str, str],
) -> Tuple[int, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    enrich: Dict[str, Any] = {}

    score = 0
    frm = norm_addr(tx.get("from"))
    to = norm_addr(tx.get("to"))
    value_eth = wei_to_eth(hex_to_int(tx.get("value", "0x0")))
    inp = tx.get("input") or "0x"
    selector = inp[:10] if inp and inp != "0x" else None

    enrich["value_eth"] = round(value_eth, 6)
    enrich["method_selector"] = selector
    enrich["method_label"] = classify_method(selector)

    # labels (context)
    if frm in labels:
        enrich["from_label"] = labels[frm]
    if to and to in labels:
        enrich["to_label"] = labels[to]

    # watchlist boost
    if frm in watchlist or (to in watchlist if to else False):
        score += 50
        reasons.append("Watchlist match")

    # high value
    if value_eth >= cfg.value_threshold_eth:
        score += 35
        reasons.append(f"High value ({value_eth:.4f} ETH ≥ {cfg.value_threshold_eth} ETH)")

    # contract creation
    if tx.get("to") is None:
        score += 30
        reasons.append("Contract creation (to=null)")

    # contract interaction
    if inp and inp != "0x":
        score += 10
        reasons.append(f"Contract interaction (selector {selector})" if selector else "Contract interaction")

    # receipt enrichments
    transfers: List[Dict[str, Any]] = []
    if receipt:
        status = receipt.get("status")
        if status == "0x0":
            score += 25
            reasons.append("Execution failed (status=0x0)")

        gas_used = hex_to_int(receipt.get("gasUsed", "0x0"))
        enrich["gas_used"] = gas_used
        if gas_used >= cfg.high_gas_threshold:
            score += 10
            reasons.append(f"High gas used ({gas_used} ≥ {cfg.high_gas_threshold})")

        transfers = parse_erc20_transfers(receipt)
        if transfers:
            score += 15
            reasons.append(f"ERC-20 Transfer events ({len(transfers)})")
            if len(transfers) >= 6:
                score += 10
                reasons.append("Burst transfer activity (≥6 transfers)")

    enrich["erc20_transfers"] = transfers
    return min(score, 100), reasons, enrich


def heuristic_summary(alert: Dict[str, Any]) -> str:
    frm = alert.get("from_label") or alert.get("from")
    to = alert.get("to_label") or alert.get("to") or "contract creation"
    value = alert.get("value_eth", 0)
    failed = any("Execution failed" in r for r in alert.get("reasons", []))
    status = "failed" if failed else "executed"
    reasons = alert.get("reasons", [])
    top = ", ".join(reasons[:3]) if reasons else "threshold met"
    rec = "Review receipt/logs in an explorer; pivot to related addresses/contracts; check for repeats."
    return f"Tx {status} from {frm} to {to} for {value} ETH. Flagged due to: {top}. Next step: {rec}"


def llm_summary(alert: Dict[str, Any], cfg: Config) -> Optional[str]:
    if not cfg.llm_endpoint or not cfg.llm_api_key:
        return None
    prompt = (
        "You are an investigations analyst. Summarize this blockchain alert in 2-3 sentences. "
        "State what happened, why it was flagged, and one recommended next step.\n\n"
        f"ALERT JSON:\n{json.dumps(alert, indent=2)}\n"
    )
    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"}
    payload = {"input": prompt, "max_output_tokens": 180}
    data = http_post_json(cfg.llm_endpoint, payload, timeout=cfg.llm_timeout_seconds, headers=headers)
    # tolerate common schemas
    return data.get("output") or data.get("text") or data.get("response")


def send_webhook(webhook_url: str, alert: Dict[str, Any]) -> None:
    if not webhook_url:
        return
    try:
        http_post_json(webhook_url, alert, timeout=10)
    except Exception:
        # demo-resilient: don't crash if webhook fails
        return


def emit_jsonl(path: str, alert: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="AI + Blockchain Integration demo bot (Sepolia)")
    ap.add_argument("--once", action="store_true", help="Scan the last N blocks and exit")
    ap.add_argument("--follow", action="store_true", help="Continuously poll for new blocks")
    args = ap.parse_args()
    if not args.once and not args.follow:
        args.once = True

    cfg = load_config()
    rpc = RpcClient(cfg.rpc_urls, timeout=30)
    watchlist = load_watchlist(cfg.watchlist_path)
    labels = load_labels(cfg.labels_path)
    conn = ensure_state_db(cfg.sqlite_state_db)

    print(f"[{utc_iso()}] chain={cfg.chain} rpc_urls={len(cfg.rpc_urls)} mode={'follow' if args.follow else 'once'}")
    print(f"[{utc_iso()}] min_score={cfg.min_score} lookback_blocks={cfg.lookback_blocks} value_threshold_eth={cfg.value_threshold_eth}")

    def process_range(start_bn: int, end_bn: int) -> None:
        receipt_fetches = 0
        for bn in range(start_bn, end_bn + 1):
            block = rpc.get_block_with_txs(bn)
            ts = hex_to_int(block.get("timestamp", "0x0"))
            block_time = utc_iso(ts)

            txs = block.get("transactions", []) or []
            for tx in txs:
                tx_hash = tx.get("hash")
                frm = norm_addr(tx.get("from"))
                to = norm_addr(tx.get("to"))
                value_eth = wei_to_eth(hex_to_int(tx.get("value", "0x0")))
                inp = tx.get("input") or "0x"

                interesting = (
                    value_eth >= cfg.value_threshold_eth
                    or inp != "0x"
                    or tx.get("to") is None
                    or frm in watchlist
                    or (to in watchlist if to else False)
                )

                receipt = None
                if interesting and receipt_fetches < cfg.max_receipt_fetch_per_block:
                    receipt = rpc.get_receipt(tx_hash)
                    receipt_fetches += 1

                score, reasons, enrich = score_transaction(tx, receipt, cfg, watchlist, labels)
                if score < cfg.min_score:
                    continue

                alert: Dict[str, Any] = {
                    "time_utc": block_time,
                    "chain": cfg.chain,
                    "block": bn,
                    "tx_hash": tx_hash,
                    "from": frm,
                    "to": to,
                    "score": score,
                    "reasons": reasons,
                    **enrich,
                }

                # AI summary layer
                if cfg.ai_enable_summary:
                    if cfg.ai_summary_mode.lower() == "llm":
                        s = llm_summary(alert, cfg)
                        if s:
                            alert["ai_summary"] = s
                        else:
                            alert["ai_summary"] = heuristic_summary(alert)
                            alert["ai_summary_note"] = "LLM not configured/failed; used heuristic summary."
                    else:
                        alert["ai_summary"] = heuristic_summary(alert)

                # outputs
                emit_jsonl(cfg.output_jsonl, alert)
                send_webhook(cfg.webhook_url, alert)

                frm_disp = alert.get("from_label") or (frm[:10] + "…" if isinstance(frm, str) else "unknown")
                to_disp = alert.get("to_label") or ((to[:10] + "…") if isinstance(to, str) else "null")
                short_reason = ", ".join(reasons[:2]) if reasons else "threshold met"
                print(f"[ALERT] bn={bn} score={score} {frm_disp} -> {to_disp} value={alert.get('value_eth')} ETH | {short_reason}")

            set_state(conn, "last_processed_block", str(bn))

    if args.once:
        latest = rpc.latest_block()
        start = max(0, latest - cfg.lookback_blocks + 1)
        print(f"[{utc_iso()}] scanning blocks {start}..{latest}")
        process_range(start, latest)
        print(f"[{utc_iso()}] done. Alerts appended to {cfg.output_jsonl}")
        return 0

    # follow mode
    while True:
        latest = rpc.latest_block()
        last = get_state(conn, "last_processed_block")
        start = max(0, latest - cfg.lookback_blocks + 1) if last is None else int(last) + 1
        if start <= latest:
            print(f"[{utc_iso()}] new blocks {start}..{latest}")
            process_range(start, latest)
        time.sleep(cfg.follow_poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
