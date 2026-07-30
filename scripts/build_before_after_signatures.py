"""Track B: before/after signatures + per-question structural opportunity.

Extends the H1 quantification to the before_after operator family using the
question's REAL anchor (not the KG-wide average). Scope: before_after questions
with an EXPLICIT DATE anchor (58.8% of them) — event anchors ("Before Ethiopia")
need event-time resolution and are deferred.

For each such question:
  * parse the anchor date + direction (before/after) from the text;
  * link topic entity + relation;
  * candidate set = all (topic, relation) facts across time (what a time-blind
    trie admits);
  * answer-verify: >=1 gold answer entity is an object of (topic, relation) on
    the correct side of the anchor;
  * per-question structural opportunity = fraction of candidate facts on the
    WRONG side of the anchor (a time-blind picker's expected temporal error).

Outputs:
  data/multitq/signatures/before_after_test.jsonl   (verified signatures, reusable for training/eval)
  results/before_after_opportunity.md
"""
import argparse
import json
import os
import pickle
import re

import re as _re
from checker.timepoint import TimePoint
from checker.constraints import resolve_before_after
from scripts.multitq_linker import MultiTQLinker, norm


def strict_norm(s: str) -> str:
    """Answer/entity normalization that KEEPS the parenthetical qualifier.

    ICEWS has many generic names distinguished only by a parenthetical
    (Military (Burundi) vs Military (Ukraine)). The lenient `norm` drops
    parentheticals and would falsely equate them, corrupting answer
    verification — so answer matching must keep them.
    """
    s = s.replace("_", " ").lower()
    s = _re.sub(r"[^a-z0-9() ]", " ", s)
    return _re.sub(r"\s+", " ", s).strip()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")
QDIR = os.path.join(ROOT, "data", "multitq", "questions")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_STOP = {"with", "that", "this", "from", "into", "about", "their", "which",
         "when", "what", "year", "time", "before", "after", "country", "who", "did"}


def parse_anchor(q):
    """Return a TimePoint anchor parsed from the question, or None."""
    t = q.lower()
    m = re.search(r"(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})", t)
    if m:
        return TimePoint(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)), "day")
    m = re.search(r"(" + "|".join(_MONTHS) + r")\s+(\d{4})", t)
    if m:
        return TimePoint(int(m.group(2)), _MONTHS[m.group(1)], 1, "month")
    m = re.search(r"\b((?:19|20)\d{2})\b", t)
    if m:
        return TimePoint(int(m.group(1)), 1, 1, "year")
    return None


def rel_keywords(s):
    return {w for w in norm(s).split() if len(w) > 3 and w not in _STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("-n", type=int, default=0)
    args = ap.parse_args()

    linker = MultiTQLinker()
    idx = pickle.load(open(IDX, "rb"))
    sr2ots = idx["sr2ots"]
    rel_kw = {r: rel_keywords(r) for r in linker.relations}

    qs = json.load(open(f"{QDIR}/{args.split}.json"))
    if args.n:
        qs = qs[:args.n]
    ba = [q for q in qs if q["qtype"] == "before_after"]

    def link_relation(question):
        qk = rel_keywords(question)
        scored = [(len(qk & rk), r) for r, rk in rel_kw.items() if rk and (qk & rk)]
        scored.sort(reverse=True)
        return [r for _, r in scored[:8]]

    verified = []
    n_explicit = 0
    opp_vals = []
    for q in ba:
        anchor = parse_anchor(q["question"])
        if anchor is None:
            continue
        n_explicit += 1
        direction = resolve_before_after(q["question"])
        topics = set(linker.link_entities(q["question"]))
        if not topics:
            continue
        rels = link_relation(q["question"])
        ans_norm = {strict_norm(a) for a in q["answers"]}
        best = None
        for s in topics:
            for r in rels:
                key = (s, r)
                if key not in sr2ots:
                    continue
                facts = sr2ots[key]  # [(ts, o)]
                # answer-verify: a gold answer is an object on the correct side.
                # Capture that (o, ts) as the training gold fact.
                gold_o = gold_ts = None
                for ts, o in facts:
                    if strict_norm(o) in ans_norm:
                        tp = TimePoint.parse(ts)
                        if (direction == "before" and tp.strictly_before(anchor)) or \
                           (direction == "after" and tp.strictly_after(anchor)):
                            gold_o, gold_ts = o, ts
                            break
                if gold_o is None:
                    continue
                # per-question opportunity: fraction of candidate facts on WRONG side
                tps = [TimePoint.parse(ts) for ts, _ in facts]
                if direction == "before":
                    wrong = sum(1 for tp in tps if not tp.strictly_before(anchor))
                else:
                    wrong = sum(1 for tp in tps if not tp.strictly_after(anchor))
                opp = wrong / len(tps)
                best = {"quid": q["quid"], "question": q["question"], "op": direction,
                        "s": s, "r": r, "o": gold_o, "gold_ts": gold_ts,
                        "anchor": str(anchor), "granularity": anchor.granularity,
                        "n_candidates": len(tps), "opportunity": round(opp, 4),
                        "answers": q["answers"]}
                break
            if best:
                break
        if best:
            verified.append(best)
            opp_vals.append(best["opportunity"])

    sig_dir = os.path.join(ROOT, "data", "multitq", "signatures")
    os.makedirs(sig_dir, exist_ok=True)
    sig_path = os.path.join(sig_dir, f"before_after_{args.split}.jsonl")
    with open(sig_path, "w") as f:
        for v in verified:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    mean_opp = sum(opp_vals) / len(opp_vals) if opp_vals else 0.0
    from collections import Counter
    by_dir = Counter(v["op"] for v in verified)
    report = [
        "# before/after — per-question structural opportunity (Track B)",
        "",
        f"- split: `{args.split}`  ·  before_after questions: {len(ba)}",
        f"- explicit-date anchor: {n_explicit} ({100*n_explicit/max(len(ba),1):.1f}%)",
        f"- **answer-verified signatures: {len(verified)} "
        f"({100*len(verified)/max(len(ba),1):.1f}% of all before_after)**  "
        f"[before {by_dir['before']} / after {by_dir['after']}]",
        f"- **mean per-question structural violation opportunity: "
        f"{100*mean_opp:.1f}%**",
        "",
        "This is the *opportunity* (time-blind candidate pollution) for before/after,",
        "now per-question with the real parsed anchor — complements the first/last",
        "realized-TVR (results/phase0_violation_report.md). The realized before/after",
        "model TVR needs a model trained on before/after paths (current model is",
        "first/last-time only) — signatures saved for that run.",
        "",
        f"Saved: `{os.path.relpath(sig_path, ROOT)}`",
    ]
    rep = os.path.join(ROOT, "results", "before_after_opportunity.md")
    open(rep, "w").write("\n".join(report))
    print("\n".join(report))
    print(f"\nwrote {sig_path} and {rep}")


if __name__ == "__main__":
    main()
