# jev-1.13 Mini Benchmark

A small, self-contained benchmark for [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13) —
TypeSafe's structured decision model, served on OpenRouter's Decisions API — on a labeled
support-triage task. 34 cases covering all three question types (`noul` / `choice` /
`score`), 4 composite cases, 5 Turkish-language cases; two full runs for consistency.
Labels are author-assigned; `clear` flags separate obvious cases from judgment calls.

Jev is not a chat model: you send a `state` plus typed questions and get calibrated
answers back — a yes/no probability (`noul`), a pick from options you define (`choice`),
or a position on an ordered rubric (`score`). No free text is generated.

## Results

| Metric | noul (urgency) | choice (department) | score (frustration) |
|---|---|---|---|
| Accuracy | 0.786 @ threshold 0.5 | 1.000 (14/14) | 1.000 within ±1; 0.857 exact |
| Clear cases | 0.917 | perfect, zero confusions | MAE 0.18 (raw scale) |
| Borderline cases | 0/2 | — | — |

- Zero false positives on urgency; negatives are confidently separated (mean p = 0.06).
- All 3 noul misses are deadline/fee cases where the model applied the stated criteria
  more literally than the labels — criteria wording is effectively the label definition.
- Consistency across runs: mean |Δp| = 0.004, max 0.04, zero decision-level flips.
- Turkish cases: no degradation (all exact or within ±1).
- Cost: $0.0011 for 68 requests (~$0.016 per 1,000 decisions); latency p50 0.41 s.

## Working principle

- **Runner/scorer split.** `run_bench.py` spends API calls and appends the full request +
  response of every call to `results/run-N.jsonl`; `score_bench.py` is offline and free
  to re-run. Raw payloads stay on disk as the audit trail.
- **Criteria as policy.** Each question's `criteria` text fixes the semantics of the
  label; `clear` flags let accuracy be read separately for obvious vs borderline cases.
- **No keys in the repo.** The runner reads the OpenRouter key from the local machine's
  auth store at runtime.

## Usage

```bash
# run (spends API calls; expects an OpenRouter key available locally)
python3 run_bench.py --tag pass1

# score (offline)
python3 score_bench.py results/run-1.jsonl [results/run-2.jsonl ...]
```

Swap `dataset.json` to benchmark a different domain — the harness (noul threshold sweep,
choice confusion matrix, score MAE, consistency check) generalizes as-is.

## Lessons learned

- **Invisible to the model catalog.** Jev is absent from `/api/v1/models`, so
  catalog-driven tools can't list or pick it; it also 400s on `chat/completions` and
  only answers on `/api/alpha/decisions`.
- **Scores are continuous**, not integers — round before thresholding.
- **Criteria wording is policy, not documentation.** The model follows it literally;
  version-control it like code.
- **Thresholds are the real product decision.** Calibrated probabilities shift the work
  to threshold selection — sweep on labeled data before pinning (here 0.3–0.5 were
  equivalent).
- **Highly deterministic** at these decision points — safe for automated gating — but
  calibrate on real, labeled data from your own domain first.
- Alpha endpoint: fine for piloting; check the direct TypeSafe API for production.
