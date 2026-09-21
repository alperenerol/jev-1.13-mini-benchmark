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

## Laya comparison (local, English cases)

Ran the same 30 English cases through [Laya](https://github.com/NandhaKishorM/laya)
(`convaiinnovations/laya` base checkpoint, Apache 2.0, self-hosted on Apple M1 Pro)
with `run_laya.py` — same dataset, same labels, two runs.

| | Laya (local, base ckpt) | Jev 1.13 (API) |
|---|---|---|
| noul accuracy @0.5 | **0.833** (10/12) | 0.750 (9/12) |
| choice accuracy | 0.750 (9/12) | **1.000** (12/12) |
| score MAE | 0.436 | **0.087** |
| score exact (rounded) | 7/12 | **12/12** |
| consistency (mean \|Δp\|) | **0.000** (fully deterministic) | 0.004 |
| latency warm | **~50 ms/case** (all questions, one forward pass) | ~400 ms/case |
| cost | **$0 self-hosted** | ~$0.016 / 1k decisions |

Reading:

- **Laya's noul is competitive** (slightly ahead on our labels) but its probabilities are
  softer (pos mean 0.73 vs Jev's 0.94) — threshold placement matters more.
- **Laya's choice weakness is real but detectable**: all 3 misses (sales-vs-other boundary,
  one technical case) came with confidence < 0.03 and flat distributions, while the worst
  confident hit sat at 0.12 — a trivial confidence gate catches every miss at the cost of
  a few escalations. Jev routed perfectly with no gating.
- **Laya's score primitive regresses to the middle** (angry → ~1.1, calm → ~1.0); the base
  checkpoint is not fine-tuned for ordinal scoring. Jev's continuous scores were near-perfect.
- **Determinism**: Laya is bit-exact across runs on the same machine; Jev drifts ±0.004.
- Both engines miss the same two borderline noul cases (trial-deadline, Thursday-deadline) —
  those are label/wording disagreements, not engine failures.

Verdict: for English triage, Jev is the better zero-shot decision engine out of the box
(choice + score especially); Laya is free, ~8× faster, bit-deterministic, self-hosted —
and its misses are gateable by its own confidence. Fine-tuning (their Kaggle recipe) is
the documented path to close the gap.

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
