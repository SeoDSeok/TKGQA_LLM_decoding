"""Answer-verified signature extraction for MultiTQ first/last-family questions.

Entity/relation linking is noisy (MultiTQ has no gold links). But for
time-answer first/last questions the *gold answer is a timestamp that must equal
the min (first) / max (last) of the target KG group*. We exploit this as a
self-verification filter: among the KG-grounded (s, r, o) signatures produced by
the noisy linker, keep only those whose group extreme matches the gold answer.
This converts noisy links into a clean, answer-consistent subset usable for
(a) exact per-question structural TVR and (b) GCR-format training paths.

Scope: qtype in {first_last, after_first, before_last} with answer_type == time
(the operators where H1 bites hardest and no external anchor is needed).

Outputs:
  data/multitq/signatures/first_last_time.jsonl   (verified signatures)
  results/signature_coverage.md
"""
import argparse
import json
import os
import pickle
import re

from checker.timepoint import TimePoint
from scripts.multitq_linker import MultiTQLinker, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")
QDIR = os.path.join(ROOT, "data", "multitq", "questions")

_STOP = {"with", "that", "this", "from", "into", "about", "their", "which",
         "when", "what", "year", "time", "first", "last", "does", "have"}


def rel_keywords(s: str) -> set:
    return {t for t in norm(s).split() if len(t) > 3 and t not in _STOP}


def is_first(question: str) -> bool:
    from checker.constraints import resolve_first_last
    return resolve_first_last(question) == "first"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("-n", type=int, default=0, help="limit questions (0=all)")
    args = ap.parse_args()

    linker = MultiTQLinker()
    idx = pickle.load(open(IDX, "rb"))
    sro2ts = idx["sro2ts"]
    rel_kw = {r: rel_keywords(r) for r in linker.relations}

    qs = json.load(open(f"{QDIR}/{args.split}.json"))
    if args.n:
        qs = qs[:args.n]
    target = [q for q in qs
              if q["qtype"] in ("first_last", "after_first", "before_last")
              and q["answer_type"] == "time"]

    def link_relation(question):
        qk = rel_keywords(question)
        scored = [(len(qk & rk), r) for r, rk in rel_kw.items() if rk and (qk & rk)]
        scored.sort(reverse=True)
        return [r for _, r in scored[:8]]

    verified = []
    n_grounded = 0
    for q in target:
        topics = set(linker.link_entities(q["question"]))
        if not topics:
            continue
        rels = link_relation(q["question"])
        try:
            ans_tp = TimePoint.parse(q["answers"][0])
        except Exception:
            continue
        want_first = is_first(q["question"])
        grounded = False
        best = None
        tl = list(topics)
        for s in tl:
            for o in tl:
                if s == o:
                    continue
                for r in rels:
                    key = (s, r, o)
                    if key not in sro2ts:
                        continue
                    grounded = True
                    ts_set = sorted({TimePoint.parse(t) for t in sro2ts[key]})
                    extreme = ts_set[0] if want_first else ts_set[-1]
                    # answer-verification: gold answer == group extreme
                    if extreme == ans_tp:
                        best = {
                            "quid": q["quid"], "question": q["question"],
                            "qtype": q["qtype"], "op": "first" if want_first else "last",
                            "s": s, "r": r, "o": o,
                            "answer": q["answers"][0],
                            "candidate_timestamps": [str(t) for t in ts_set],
                            "n_candidates": len(ts_set),
                        }
                        break
                if best:
                    break
            if best:
                break
        if grounded:
            n_grounded += 1
        if best:
            verified.append(best)

    # write
    sig_dir = os.path.join(ROOT, "data", "multitq", "signatures")
    os.makedirs(sig_dir, exist_ok=True)
    sig_path = os.path.join(sig_dir, f"first_last_time_{args.split}.jsonl")
    with open(sig_path, "w") as f:
        for v in verified:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    nt = len(target)
    multi = sum(1 for v in verified if v["n_candidates"] >= 2)
    report = [
        "# MultiTQ signature coverage (answer-verified, first/last time-answers)",
        "",
        f"- split: `{args.split}`",
        f"- target questions (first/last-family, answer_type=time): **{nt}**",
        f"- KG-grounded (≥1 signature): {n_grounded} ({100*n_grounded/max(nt,1):.1f}%)",
        f"- **answer-verified clean signatures: {len(verified)} "
        f"({100*len(verified)/max(nt,1):.1f}%)**",
        f"- of those, with ≥2 candidate timestamps (real first/last 'trap'): "
        f"{multi} ({100*multi/max(len(verified),1):.1f}%)",
        "",
        "Answer-verification: kept only signatures whose group min(first)/max(last)",
        "timestamp equals the gold answer — this filters out the noisy linker's",
        "false positives, yielding a clean subset for training + exact TVR.",
        "",
        f"Saved: `{os.path.relpath(sig_path, ROOT)}`",
    ]
    rep_path = os.path.join(ROOT, "results", "signature_coverage.md")
    open(rep_path, "w").write("\n".join(report))
    print("\n".join(report))
    print(f"\nwrote {sig_path} and {rep_path}")


if __name__ == "__main__":
    main()
