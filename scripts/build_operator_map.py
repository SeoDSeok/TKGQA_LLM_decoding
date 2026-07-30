"""Emit the MultiTQ qtype -> temporal-operator mapping table (deliverable).

Joins the operator taxonomy (checker.constraints.MULTITQ_QTYPE_MAP) with the
actual per-split question counts, and writes both a machine-readable JSON
(configs/multitq_operator_map.json) and a human-readable markdown table
(results/operator_map.md).  This is the "질문 유형 <-> 시간 연산자 매핑 테이블"
Phase 0 checklist item, used for the per-operator TVR breakdown (H2).
"""
import json
import os
from collections import Counter

from checker.constraints import MULTITQ_QTYPE_MAP, COMPLEXITY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_DIR = os.path.join(ROOT, "data", "multitq", "questions")

# Which atomic operators each qtype exercises (for the H2 complexity axis).
QTYPE_OPERATORS = {
    "equal": ["equal"],
    "equal_multi": ["equal", "multi"],
    "before_after": ["before", "after"],
    "first_last": ["first", "last"],
    "after_first": ["after", "first"],
    "before_last": ["before", "last"],
}


def qtype_complexity(qtype: str) -> int:
    return max(COMPLEXITY[op] for op in QTYPE_OPERATORS[qtype])


def load_counts():
    counts = {}
    for split in ("train", "dev", "test"):
        p = os.path.join(Q_DIR, f"{split}.json")
        if not os.path.exists(p):
            counts[split] = {}
            continue
        data = json.load(open(p))
        counts[split] = dict(Counter(q["qtype"] for q in data))
    return counts


def build():
    counts = load_counts()
    rows = []
    for qtype, tmpl in MULTITQ_QTYPE_MAP.items():
        rows.append({
            "qtype": qtype,
            "primary_operator": tmpl["primary"],
            "interval_op": tmpl["interval_op"],
            "ordering": tmpl["ordering"],
            "operators": QTYPE_OPERATORS[qtype],
            "complexity": qtype_complexity(qtype),
            "note": tmpl["note"],
            "counts": {s: counts.get(s, {}).get(qtype, 0) for s in ("train", "dev", "test")},
        })
    rows.sort(key=lambda r: (r["complexity"], r["qtype"]))
    return rows


def write_json(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"multitq_operator_map": rows}, open(path, "w"), indent=2, ensure_ascii=False)


def write_md(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# MultiTQ qtype → Temporal Operator Map",
        "",
        "Evaluation-only labelling for the TVR metric and the H2 per-operator",
        "breakdown (complexity: simple < before/after < first/last < multi).",
        "before/after and first/last are resolved from the question surface form",
        "(see `checker/constraints.py::build_constraint`).",
        "",
        "| qtype | primary op | interval filter | ordering | complexity | train | dev | test | note |",
        "|---|---|---|---|:-:|--:|--:|--:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['qtype']}` | {r['primary_operator']} | {r['interval_op'] or '—'} | "
            f"{r['ordering'] or '—'} | {r['complexity']} | "
            f"{r['counts']['train']} | {r['counts']['dev']} | {r['counts']['test']} | {r['note']} |"
        )
    totals = {s: sum(r["counts"][s] for r in rows) for s in ("train", "dev", "test")}
    lines.append(f"| **total** | | | | | {totals['train']} | {totals['dev']} | {totals['test']} | |")
    lines.append("")
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    rows = build()
    write_json(rows, os.path.join(ROOT, "configs", "multitq_operator_map.json"))
    write_md(rows, os.path.join(ROOT, "results", "operator_map.md"))
    print("wrote configs/multitq_operator_map.json and results/operator_map.md")
    for r in rows:
        print(f"  {r['qtype']:14s} -> {r['primary_operator']:7s} (complexity {r['complexity']}) "
              f"test={r['counts']['test']}")
