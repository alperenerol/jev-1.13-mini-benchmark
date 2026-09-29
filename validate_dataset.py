#!/usr/bin/env python3
"""Validate dataset.json for jev-bench.

Default mode (and `--dry-run`, accepted as a no-op alias) validates the
dataset and prints stats. ZERO writes happen in this mode. The only write
path is an explicit `--output PATH`, which additionally writes a validation
report to that path.

Exit codes: 0 = valid dataset; 1 = missing file, invalid JSON, or failed
validation.

Usage:
    python3 validate_dataset.py [--dataset PATH] [--output PATH] [--dry-run]
"""

import argparse
import json
import sys
from collections import Counter

DEFAULT_DATASET = "dataset.json"

REQUIRED_CASE_FIELDS = ("id", "lang", "state", "questions", "expected")


def load_dataset(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"dataset file not found: {path}"
    except json.JSONDecodeError as e:
        return None, f"invalid JSON in {path}: {e}"
    except OSError as e:
        return None, f"cannot read {path}: {e}"


def validate_label(spec, value):
    """Return an error string if `value` is not a valid label for spec, else None."""
    stype = spec.get("type")
    criteria = spec.get("criteria")
    if stype == "choice":
        if not (isinstance(value, str) and value in (criteria or {})):
            return f"label {value!r} not a valid choice ({sorted(criteria or {})})"
    elif stype == "noul":
        if not isinstance(value, bool):
            return f"label {value!r} not a boolean (true/false required)"
    elif stype == "score":
        n = len(criteria) if isinstance(criteria, (list, dict)) else 0
        if not (isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value < max(n, 1)):
            return f"label {value!r} not an integer score in [0, {max(n - 1, 0)}]"
    else:
        return f"unknown spec type {stype!r}"
    return None


def validate(dataset, dataset_path):
    """Validate the parsed dataset. Returns (issues, stats)."""
    issues = []

    if not isinstance(dataset, dict):
        return [f"{dataset_path}: top-level structure must be a JSON object"], {}
    for field in ("name", "version", "question_specs", "cases"):
        if field not in dataset:
            issues.append(f"missing top-level field: {field}")
    if issues:
        return issues, {}

    specs = dataset["question_specs"]
    cases = dataset["cases"]
    if not isinstance(specs, dict):
        issues.append("question_specs must be an object mapping spec name -> spec")
        specs = {}
    if not isinstance(cases, list):
        issues.append("cases must be an array")
        cases = []

    # Per-case schema validation
    seen_ids = {}
    lang_counts = Counter()
    label_dist = {}  # spec name -> Counter of labels
    for i, case in enumerate(cases):
        where = f"case[{i}]"
        if not isinstance(case, dict):
            issues.append(f"{where}: case must be an object")
            continue
        cid = case.get("id")
        if isinstance(cid, str) and cid:
            where = f"case {cid}"
            seen_ids[cid] = seen_ids.get(cid, 0) + 1

        for field in REQUIRED_CASE_FIELDS:
            v = case.get(field)
            if v is None or (isinstance(v, str) and not v.strip()):
                issues.append(f"{where}: missing/empty required field {field!r}")

        lang = case.get("lang")
        if isinstance(lang, str) and lang:
            lang_counts[lang] += 1

        questions = case.get("questions")
        expected = case.get("expected")
        if not isinstance(questions, dict) or not questions:
            issues.append(f"{where}: 'questions' must be a non-empty object")
            questions = {}
        if not isinstance(expected, dict):
            issues.append(f"{where}: 'expected' must be an object")
            expected = {}

        for qname, _q in questions.items():
            if qname not in specs:
                issues.append(f"{where}: question {qname!r} not defined in question_specs")
                if qname not in expected:
                    issues.append(f"{where}: no expected value for question {qname!r}")
                continue
            exp = expected.get(qname)
            if not isinstance(exp, dict):
                issues.append(f"{where}: expected value for {qname!r} must be an object")
                continue
            if "label" not in exp or exp["label"] in (None, ""):
                issues.append(f"{where}: {qname}: missing/empty 'label'")
            else:
                err = validate_label(specs[qname], exp["label"])
                if err:
                    issues.append(f"{where}: {qname}: {err}")
                label_dist.setdefault(qname, Counter())[str(exp["label"])] += 1
            if "clear" not in exp or not isinstance(exp["clear"], bool):
                issues.append(f"{where}: {qname}: 'clear' must be present and boolean")
        for qname in expected:
            if qname not in questions:
                issues.append(f"{where}: expected entry {qname!r} has no matching question")

    dups = {cid: n for cid, n in seen_ids.items() if n > 1}
    for cid, n in sorted(dups.items()):
        issues.append(f"duplicate id: {cid!r} appears {n} times")

    stats = {
        "entry_count": len(cases),
        "languages": dict(sorted(lang_counts.items())),
        "labels_by_spec": {k: dict(v.most_common()) for k, v in sorted(label_dist.items())},
        "duplicate_ids": dups,
    }
    return issues, stats


def format_stats(stats, issues):
    lines = []
    lines.append(f"entries: {stats.get('entry_count', 0)}")
    langs = stats.get("languages", {})
    total = sum(langs.values()) or 1
    lines.append("language distribution:")
    for lang, n in langs.items():
        lines.append(f"  {lang}: {n} ({n * 100 / total:.1f}%)")
    lines.append("label distribution (by question spec):")
    for spec, counts in stats.get("labels_by_spec", {}).items():
        joined = ", ".join(f"{label}={n}" for label, n in counts.items())
        lines.append(f"  {spec}: {joined}")
    dups = stats.get("duplicate_ids", {})
    lines.append("duplicate ids: " + (", ".join(sorted(dups)) if dups else "none"))
    lines.append(f"schema/field issues: {len(issues)}")
    for issue in issues:
        lines.append(f"  - {issue}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="validate_dataset.py",
        description="Validate jev-bench dataset.json and print stats. "
                    "Dry-run by default (zero writes); writes a report only with --output.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"path to dataset file (default: {DEFAULT_DATASET})")
    parser.add_argument("--output", default=None,
                        help="optional path to write the validation report (only write path)")
    parser.add_argument("--dry-run", action="store_true",
                        help="no-op alias; validation is dry-run by default")
    args = parser.parse_args(argv)

    dataset, err = load_dataset(args.dataset)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    issues, stats = validate(dataset, args.dataset)
    report = format_stats(stats, issues)

    print(report)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report + "\n")
            print(f"report written to {args.output}")
        except OSError as e:
            print(f"error: cannot write report to {args.output}: {e}", file=sys.stderr)
            return 1

    if issues:
        print(f"FAILED: {len(issues)} issue(s) found in {args.dataset}", file=sys.stderr)
        return 1
    print(f"OK: {args.dataset} is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())