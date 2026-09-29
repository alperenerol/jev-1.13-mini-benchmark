#!/usr/bin/env python3
"""Scorer: grades run JSONL files against dataset labels. Never calls the API.

Metric design
- Per question, graded by its spec type (from dataset.json question_specs):
  * noul: hit if probability crosses the decision threshold toward the labeled side.
    Threshold sweep reported (0.3..0.7); primary operating point 0.5. Also ROC-ish stats
    (best accuracy threshold, discrimination between label groups).
  * choice: hit if argmax choice == label; report per-class accuracy + confusion pairs.
  * score: hit if |score - label| <= 1; exact-match reported separately.
- Composite cases graded per-question; clear vs borderline breakdown for every type.
- Report is keyed per question name; each entry carries its spec type.
Usage:
  python3 score_bench.py results/run-1.jsonl [results/run-2.jsonl ...]
  python3 score_bench.py --validate            # dataset structural validation, no runs
"""
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_dataset():
    ds = json.loads((HERE / "dataset.json").read_text())
    labels = {}
    for case in ds["cases"]:
        labels[case["id"]] = {
            "lang": case.get("lang", "en"),
            "expected": {q: spec["label"] for q, spec in case["expected"].items()},
            "clear": {q: spec["clear"] for q, spec in case["expected"].items()},
        }
    return ds, labels


def extract_answers(row):
    resp = row.get("response")
    if not isinstance(resp, dict):
        return None
    return resp.get("answers")


# ---------- per-type graders (parameterized by question name) ----------

def grade_noul(rows, labels, qname):
    """rows: list of (row, answers). Returns metrics dict."""
    pts = []
    for row, ans in rows:
        a = ans.get(qname)
        if not isinstance(a, dict) or "noul" not in a:
            continue
        label = labels[row["case_id"]]["expected"][qname]
        pts.append((row["case_id"], float(a["noul"]), 1 if label else 0,
                    labels[row["case_id"]]["clear"][qname]))
    if not pts:
        return {"count": 0}
    pts.sort(key=lambda x: x[1], reverse=True)
    sweep = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        acc = sum(1 for _, p, y, _ in pts if (p >= t) == (y == 1)) / len(pts)
        sweep[t] = round(acc, 3)
    best_t = max(sweep, key=sweep.get)
    pos = [p for _, p, y, _ in pts if y == 1]
    neg = [p for _, p, y, _ in pts if y == 0]
    def acc_at(t):
        return sum(1 for _, p, y, _ in pts if (p >= t) == (y == 1)) / len(pts)
    # confusion at 0.5
    fp = [cid for cid, p, y, _ in pts if y == 0 and p >= 0.5]
    fn = [cid for cid, p, y, _ in pts if y == 1 and p < 0.5]
    clear_acc = [x for x in pts if x[3]]
    bord_acc = [x for x in pts if not x[3]]
    return {
        "count": len(pts),
        "accuracy@0.5": round(acc_at(0.5), 3),
        "threshold_sweep": sweep,
        "best_threshold": best_t,
        "pos_mean": round(sum(pos) / len(pos), 3) if pos else None,
        "neg_mean": round(sum(neg) / len(neg), 3) if neg else None,
        "fp@0.5": fp, "fn@0.5": fn,
        "clear_acc@0.5": round(sum(1 for _, p, y, _ in clear_acc if (p >= 0.5) == (y == 1)) / len(clear_acc), 3) if clear_acc else None,
        "borderline_acc@0.5": round(sum(1 for _, p, y, _ in bord_acc if (p >= 0.5) == (y == 1)) / len(bord_acc), 3) if bord_acc else None,
        "sorted_probs": [(cid, round(p, 3), y) for cid, p, y, _ in pts],
    }


def grade_choice(rows, labels, qname):
    per_class = defaultdict(lambda: [0, 0])  # label -> [hits, total]
    confusions = defaultdict(int)
    misses = []
    n = hits = 0
    for row, ans in rows:
        exp = labels[row["case_id"]]
        a = ans.get(qname)
        if not isinstance(a, dict) or "choice" not in a:
            continue
        n += 1
        label = exp["expected"][qname]
        got = a["choice"]
        per_class[label][1] += 1
        if got == label:
            hits += 1
            per_class[label][0] += 1
        else:
            confusions[f"{label}->{got}"] += 1
            misses.append({"case": row["case_id"], "label": label, "got": got,
                           "probs": a.get("probabilities")})
    return {"count": n, "accuracy": round(hits / n, 3) if n else None,
            "per_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_class.items())},
            "confusions": dict(confusions), "misses": misses}


def grade_score(rows, labels, qname):
    n = hits = exact = 0
    abs_errs = []
    dist = defaultdict(int)
    miss_examples = []
    for row, ans in rows:
        a = ans.get(qname)
        if not isinstance(a, dict) or "score" not in a:
            continue
        n += 1
        label = labels[row["case_id"]]["expected"][qname]
        raw = float(a["score"])
        got = int(round(raw))  # API returns continuous scores in [0, len(criteria)-1]
        abs_errs.append(abs(raw - label))
        dist[got - label] += 1
        if got == label:
            exact += 1
        if abs(got - label) <= 1:
            hits += 1
        elif len(miss_examples) < 5:
            miss_examples.append({"case": row["case_id"], "label": label, "got": raw})
    mae = round(sum(abs_errs) / len(abs_errs), 3) if abs_errs else None
    return {"count": n, "accuracy_within_1": round(hits / n, 3) if n else None,
            "exact_match_rounded": round(exact / n, 3) if n else None,
            "mae_raw": mae,
            "error_distribution_rounded": dict(sorted(dist.items())), "miss_examples": miss_examples}


GRADERS = {"noul": grade_noul, "choice": grade_choice, "score": grade_score}


def score_run(path: Path, ds, labels) -> dict:
    rows_by_case = defaultdict(list)
    errors = []
    excluded = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["case_id"] not in labels:
            excluded.add(row["case_id"])  # result predates a dataset edit; skip cleanly
            continue
        if row["http_status"] != 200:
            errors.append(row["case_id"])
            continue
        ans = extract_answers(row)
        if ans is None:
            errors.append(row["case_id"])
            continue
        rows_by_case[row["case_id"]].append((row, ans))

    # if a case appears multiple times in one file (consistency re-run), grade each copy
    report = {"file": path.name, "api_errors": errors, "excluded_cases": sorted(excluded)}

    all_rows = [x for pair in rows_by_case.items() for x in pair[1]]
    # grade every question name in the dataset by its spec type, keyed per question
    report["questions"] = {}
    for qname, spec in ds["question_specs"].items():
        q_rows = [x for x in all_rows if qname in x[1]]
        report["questions"][qname] = {"type": spec["type"],
                                      **GRADERS[spec["type"]](q_rows, labels, qname)}

    # consistency: same case asked twice in the same run
    dup = {cid: v for cid, v in rows_by_case.items() if len(v) > 1}
    if dup:
        flips = []
        for cid, lst in dup.items():
            for qname in lst[0][1]:
                vals = []
                for _, a in lst:
                    q = a.get(qname)
                    if not isinstance(q, dict):
                        continue
                    if "noul" in q:
                        vals.append(round(q["noul"], 2))
                    elif "choice" in q:
                        vals.append(q["choice"])
                    elif "score" in q:
                        vals.append(q["score"])
                if len(set(vals)) > 1:
                    flips.append({"case": cid, "question": qname, "values": vals})
        report["consistency"] = {"repeat_cases": list(dup), "flips": flips}
    return report


# ---------- dataset structural validation ----------

def validate_dataset(ds) -> tuple:
    problems = []
    specs = ds.get("question_specs", {})
    ids = set()
    for spec_name, spec in specs.items():
        if spec.get("type") not in ("noul", "choice", "score"):
            problems.append(f"spec {spec_name}: unknown type {spec.get('type')!r}")
        elif spec.get("type") == "choice" and not isinstance(spec.get("criteria"), dict):
            problems.append(f"spec {spec_name}: choice criteria must be a dict of options")
        elif spec.get("type") == "score" and not isinstance(spec.get("criteria"), list):
            problems.append(f"spec {spec_name}: score criteria must be a list")
        elif spec.get("type") == "noul" and not isinstance(spec.get("criteria"), dict):
            problems.append(f"spec {spec_name}: noul criteria must be a dict")
    for case in ds["cases"]:
        cid = case.get("id")
        if not cid:
            problems.append("case without id")
            continue
        if cid in ids:
            problems.append(f"duplicate id: {cid}")
        ids.add(cid)
        lang = case.get("lang")
        if lang not in ("en", "tr"):
            problems.append(f"{cid}: lang {lang!r} not in {{en,tr}}")
        qs = case.get("questions", {})
        if not qs:
            problems.append(f"{cid}: no questions")
        for qname in qs:
            if qname not in specs:
                problems.append(f"{cid}: question {qname!r} has no spec")
            elif qs[qname] != "spec":
                problems.append(f"{cid}: question {qname!r} value must be 'spec'")
        exp = case.get("expected", {})
        if set(exp) != set(qs):
            problems.append(f"{cid}: expected keys {sorted(exp)} != questions keys {sorted(qs)}")
        for qname, e in exp.items():
            if qname not in specs:
                continue
            stype = specs[qname]["type"]
            if not isinstance(e.get("clear"), bool):
                problems.append(f"{cid}/{qname}: clear flag not boolean")
            label = e.get("label")
            if stype == "noul":
                if not isinstance(label, bool):
                    problems.append(f"{cid}/{qname}: noul label {label!r} not boolean")
            elif stype == "choice":
                if label not in specs[qname]["criteria"]:
                    problems.append(f"{cid}/{qname}: choice label {label!r} not in options")
            elif stype == "score":
                ncrit = len(specs[qname]["criteria"])
                if not isinstance(label, int) or not (0 <= label <= ncrit - 1):
                    problems.append(f"{cid}/{qname}: score label {label!r} out of range 0..{ncrit-1}")
    lang_counts = dict(Counter(c.get("lang") for c in ds["cases"]))
    return problems, lang_counts


def main():
    args = sys.argv[1:]
    if "--validate" in args:
        args.remove("--validate")
        ds, _ = load_dataset()
        problems, lang_counts = validate_dataset(ds)
        print(json.dumps({"cases": len(ds["cases"]),
                          "lang_counts": lang_counts,
                          "problems": problems,
                          "ok": not problems}, indent=1, ensure_ascii=False))
        sys.exit(0 if not problems else 2)
    if not args:
        print(__doc__)
        sys.exit(1)
    ds, labels = load_dataset()
    for arg in args:
        p = Path(arg)
        if not p.exists():
            print(f"missing: {p}")
            continue
        rep = score_run(p, ds, labels)
        print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
