#!/usr/bin/env python3
"""Runner: calls OpenRouter /api/alpha/decisions for every dataset case.

- Reads the OpenRouter key from ~/.hermes/auth.json credential_pool (never prints it).
- Appends raw API responses to results/run-N.jsonl (one line per request, full payload).
- Idempotent per run dir; re-run creates a new run-N.
- Bounded retries with backoff on 429/5xx; a case that still fails is recorded as an error row,
  never silently skipped.
Usage: python3 run_bench.py [--tag LABEL]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
MAX_RETRIES = 4
TIMEOUT_S = 60


def load_key() -> str:
    import os
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env
    auth = json.loads((Path.home() / ".hermes" / "auth.json").read_text())
    for cred in auth.get("credential_pool", {}).get("openrouter", []):
        if isinstance(cred, dict) and cred.get("access_token"):
            return cred["access_token"]
    sys.exit("Set OPENROUTER_API_KEY (or configure a key in the local auth store)")


def call_api(key: str, payload: dict) -> tuple[int, dict | str]:
    """POST one decision request. Returns (status, parsed-json-or-error-string)."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/jev-bench",
            "X-OpenRouter-Title": "jev-mini-bench",
        },
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
            except Exception:
                body = {"raw": "undecodable"}
            # retry only on transient conditions
            if e.code == 429 or 500 <= e.code < 600:
                wait = min(2 ** attempt * 2, 30)
                print(f"  retry {attempt}/{MAX_RETRIES} after HTTP {e.code}, sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            return e.code, body  # permanent client error
        except (urllib.error.URLError, TimeoutError) as e:
            wait = min(2 ** attempt * 2, 30)
            print(f"  network error ({e}); retry {attempt}/{MAX_RETRIES} in {wait}s", flush=True)
            time.sleep(wait)
    return 0, "exhausted retries"


def build_questions(specs: dict, case: dict) -> dict:
    qs = {}
    for qname, tag in case["questions"].items():
        if tag != "spec":
            qs[qname] = tag  # inline override
            continue
        spec = specs[qname]
        entry = {"type": spec["type"], "instructions": spec["instructions"]}
        crit = spec.get("criteria")
        if isinstance(crit, list):
            entry["criteria"] = list(crit)
        else:
            entry["criteria"] = dict(crit)
        qs[qname] = entry
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="optional label recorded in each row")
    args = ap.parse_args()

    dataset = json.loads((HERE / "dataset.json").read_text())
    specs = dataset["question_specs"]
    run_dir = HERE / "results"
    run_dir.mkdir(exist_ok=True)
    n = 1
    while (run_dir / f"run-{n}.jsonl").exists():
        n += 1
    out_path = run_dir / f"run-{n}.jsonl"

    key = load_key()
    print(f"run-{n}: {len(dataset['cases'])} cases -> {out_path.name}", flush=True)

    ok = err = 0
    started = time.time()
    with out_path.open("a") as out:
        for case in dataset["cases"]:
            payload = {
                "model": MODEL,
                "state": case["state"],
                "questions": build_questions(specs, case),
            }
            t0 = time.time()
            status, resp = call_api(key, payload)
            elapsed = round(time.time() - t0, 2)
            row = {
                "case_id": case["id"],
                "run": n,
                "tag": args.tag,
                "http_status": status,
                "elapsed_s": elapsed,
                "request": payload,   # full request for auditability
                "response": resp,     # full response (or error body)
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if status == 200 and isinstance(resp, dict) and "answers" in resp:
                ok += 1
                print(f"  {case['id']:<8} 200 {elapsed:>5}s", flush=True)
            else:
                err += 1
                print(f"  {case['id']:<8} HTTP {status} ERROR: {str(resp)[:120]}", flush=True)
            time.sleep(0.4)  # be gentle with rate limits

    dt = round(time.time() - started, 1)
    print(f"done: {ok} ok, {err} errored, {dt}s total -> {out_path}")
    sys.exit(0 if err == 0 else 1)


if __name__ == "__main__":
    main()
