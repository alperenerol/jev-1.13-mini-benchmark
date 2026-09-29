# jev-1.13 Mini Benchmark

A small, self-contained benchmark for [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13) —
TypeSafe's structured decision model, served on OpenRouter's Decisions API — on a labeled
support-triage task. Dataset v1.2: 100 cases (96 English / 4 Turkish) covering all three
core question types (`noul` / `choice` / `score`), composite cases, and — new in v1.2 —
68 agentic-routing cases (`rt_*`, cases 33–100). Cases 33–100 are **agentic-routing and
English-only**: they exercise a six-way routing decision (`route_target`), an escalation
probability (`escalate`), and a self-assessed route confidence (`route_confidence`).
Two full runs per model for consistency.
Labels are author-assigned; `clear` flags separate obvious cases from judgment calls.
(v1.1: two contestable deadline cases were removed — the "misses" there were wording
disputes about the label, not model errors; all scores below are recomputed on the
revised set. v1.2 adds the 68 routing cases and per-question scorer output.)

Jev is not a chat model: you send a `state` plus typed questions and get calibrated
answers back — a yes/no probability (`noul`), a pick from options you define (`choice`),
or a position on an ordered rubric (`score`). No free text is generated.

## Results

| Metric | noul (urgency) | choice (department) | score (frustration) |
|---|---|---|---|
| Accuracy | 0.917 @ threshold 0.5 (11/12) | 1.000 (13/13) | 1.000 within ±1; 0.846 exact |
| Clear cases | 1.000 | perfect, zero confusions | MAE 0.20 (raw scale) |
| Borderline cases | 0/1 | — | — |

- Zero false positives on urgency; negatives are confidently separated (mean p = 0.099).
- The single noul miss (noul_03, a double-charge refund) is a borderline case — the model
  applied the stated criteria more literally than the label; criteria wording is
  effectively the label definition.
- Consistency across runs: mean |Δp| = 0.004, max 0.04, zero decision-level flips.
- Turkish cases: no degradation (all exact or within ±1).
- Cost: ~$0.016 per 1,000 decisions ($0.0011 for the 68-request benchmark); latency p50 0.41 s.

## Agentic-routing results (v1.2, cases 33–100)

The 68 new `rt_*` cases (English-only) add three questions on top of the legacy set:
`route_target` (choice over 6 fixed routes), `escalate` (a yes/no escalation-likelihood
probability), and `route_confidence` (score 0–2). Two full runs
per model: Jev `run-3`/`run-4`, Laya `laya-run-3`/`laya-run-4` (96 English cases; 4
Turkish cases excluded by `run_laya.py` design).

### Jev vs Laya, per question type (100-case set, 2 runs each)

| Metric | Jev 1.13 (API) | Laya (local, base ckpt) |
|---|---|---|
| is_urgent (noul) acc @0.5 | 0.917 (12 legacy cases) | 1.000 (10 en cases) |
| escalate acc @0.5 | **0.95** (19/20); 0.3–0.6 sweep all 0.95 | 0.70 @0.5; best 0.80 @0.7 |
| escalate clear / borderline | 1.000 / 0.75 | 0.812 / 0.25 |
| route_target acc (56 rt cases) | **0.607** (34/56), both runs | 0.196 (11/56), both runs |
| route_target best classes | escalation_human 12/12, technical_specialist 3/3, billing 8/9 | self_handle 7/10 only; triage_agent 0/12, technical 0/3 |
| route_target weakest classes | refuse_out_of_scope 0/10, self_handle 2/10 | collapses to self_handle / refuse_out_of_scope |
| route_confidence within-1 | **0.975** both runs | 0.90 both runs |
| route_confidence exact / MAE | 0.30–0.375 / 0.61–0.62 (skews confident) | 0.40 / 0.631 (4 large overshoots: labeled 0, scored 1.58–1.75) |

Top route_target confusions (Jev): refuse_out_of_scope→self_handle ×5,
self_handle→billing_specialist ×4, refuse_out_of_scope→escalation_human ×3–4.
Laya cannot separate triage_agent vs self_handle at all (9/12 triage_agent→self_handle).

### Cross-run consistency

- **Jev**: 0.990 — 99/100 identical decisions across run-3/run-4; sole flip is
  rt_oos_03 route_target (escalation_human ↔ billing_specialist, a genuinely borderline
  case). Probability drift mean |Δp| ≈ 0.005.
- **Laya**: 1.000 — bit-identical across both runs (deterministic local forward pass).

### Cost and latency

| | Jev 1.13 (API) | Laya (local) |
|---|---|---|
| cost per full 100-case run | $0.002055 (identical both runs) | $0 self-hosted |
| cost per case | ~$0.0000206 (~48.9k in / ~6.6k out tokens) | $0 |
| latency mean / max | 0.35–0.37 s / 0.92 s | 0.09–0.10 s / 1.29 s (model load) |
| wall time per run | ~35–37 s | ~9–10 s |

Reading: on the new agentic-routing surface Jev remains clearly ahead — 3× better
route_target accuracy and much better-separated escalate probabilities (pos 0.863 vs
neg 0.189; Laya's pos 0.664 vs neg 0.444 means the threshold choice dominates: Jev is
flat 0.95 across 0.3–0.6, Laya needs 0.7). Jev's one weak area is refusing
out-of-scope requests (0/10 recall) — it routes them to self_handle or escalation
instead. Raw run payloads: `results/run-3.jsonl`, `run-4.jsonl`, `laya-run-3.jsonl`,
`laya-run-4.jsonl` with `.score.json` reports alongside.

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
| noul accuracy @0.5 | **1.000** (10/10) | 0.900 (9/10) |
| choice accuracy | 0.818 (9/11) | **1.000** (11/11) |
| score MAE | 0.441 | **0.094** |
| score exact (rounded) | 6/11 | **11/11** |
| consistency (mean \|Δp\|) | **0.000** (fully deterministic) | 0.004 |
| latency warm | **~50 ms/case** (all questions, one forward pass) | ~400 ms/case |
| cost | **$0 self-hosted** | ~$0.016 / 1k decisions |

Reading:

- **Laya's noul is clean on the revised set** (10/10) but its probabilities are softer
  (pos mean 0.73 vs Jev's 0.94) — threshold placement matters more.
- **Laya's choice weakness is real but detectable**: both misses (sales-vs-other boundary,
  usage-vs-technical) came with confidence < 0.03 and flat distributions, while the worst
  confident hit sat at 0.12 — a trivial confidence gate catches every miss at the cost of
  a few escalations. Jev routed perfectly with no gating.
- **Laya's score primitive regresses to the middle** (angry → ~1.1, calm → ~1.0); the base
  checkpoint is not fine-tuned for ordinal scoring. Jev's continuous scores were near-perfect.
- **Determinism**: Laya is bit-exact across runs on the same machine; Jev drifts ±0.004.

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

## Dataset validation

Validate `dataset.json` (dry-run by default — validates and prints stats, writes nothing):

    python3 validate_dataset.py

Options: `--dataset PATH` (default `dataset.json`), `--output PATH` (write the
validation report to a file — the only write path), `--dry-run` (no-op alias).
Exit code 0 = valid dataset; 1 = missing file, invalid JSON, or validation
failure (missing/empty required fields, duplicate ids, labels outside the
question-spec criteria).
