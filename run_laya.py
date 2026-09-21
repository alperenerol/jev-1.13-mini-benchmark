#!/usr/bin/env python3
"""Laya adapter for jev-bench: runs the same dataset through the local laya model.

- English cases only (lang == 'en') — per scope decision.
- One agent.predict(state, questions) per case (all questions answered in a single
  forward pass — this is Laya's non-autoregressive trick, vs one API call per case on Jev).
- Rows append to results/laya-run-N.jsonl in the jev-bench schema: http_status=200 means
  the local forward pass succeeded (kept so score_bench.py grades both engines alike).
Usage: python3 run_laya.py [--tag LABEL] [--device mps|cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="optional label recorded in each row")
    ap.add_argument("--device", default=None, help="mps or cpu (default: laya's own choice)")
    args = ap.parse_args()

    import laya  # imported late so --help works without torch

    dataset = json.loads((HERE / "dataset.json").read_text())
    specs = dataset["question_specs"]
    cases = [c for c in dataset["cases"] if c.get("lang", "en") == "en"]
    print(f"English cases: {len(cases)} of {len(dataset['cases'])} (Turkish excluded)", flush=True)

    run_dir = HERE / "results"
    run_dir.mkdir(exist_ok=True)
    n = 1
    while (run_dir / f"laya-run-{n}.jsonl").exists():
        n += 1
    out_path = run_dir / f"laya-run-{n}.jsonl"

    t0 = time.time()
    agent = laya.load("convaiinnovations/laya")
    if args.device:
        try:
            agent.model.to(args.device)  # Agent wraps a torch model; move it explicitly
        except AttributeError:
            print(f"note: could not set device {args.device!r} on this agent object; using default", flush=True)
    print(f"model loaded in {round(time.time() - t0, 1)}s", flush=True)

    ok = err = 0
    total_fwd = 0.0
    with out_path.open("a") as out:
        for case in cases:
            questions = {}
            for qname, tag in case["questions"].items():
                spec = specs[qname]
                entry = {"type": spec["type"], "instructions": spec["instructions"]}
                crit = spec.get("criteria")
                entry["criteria"] = list(crit) if isinstance(crit, list) else dict(crit)
                questions[qname] = entry

            t1 = time.time()
            try:
                res = agent.predict(case["state"], questions)
                status, resp = 200, res
                ok += 1
            except Exception as e:  # noqa: BLE001 - record and continue
                status, resp = 0, {"error": str(e)}
                err += 1
            elapsed = round(time.time() - t1, 3)
            total_fwd += elapsed

            row = {
                "case_id": case["id"],
                "run": n,
                "tag": args.tag,
                "engine": "laya",
                "http_status": status,
                "elapsed_s": elapsed,
                "request": {"model": "convaiinnovations/laya", "state": case["state"], "questions": questions},
                "response": resp,
            }
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            out.flush()
            mark = "ok " if status == 200 else "ERR"
            print(f"  {case['id']:<8} {mark} {elapsed:>6}s", flush=True)

    print(f"done: {ok} ok, {err} errored | forward-pass total {round(total_fwd,1)}s | -> {out_path}")
    sys.exit(0 if err == 0 else 1)


if __name__ == "__main__":
    main()
