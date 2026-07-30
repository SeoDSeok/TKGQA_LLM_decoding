"""Phase-0 headroom gate for the KBS gap-closing plan.

Reads TimeR4's RELEASED MultiTQ predictions (results.zip -> predictions.jsonl),
whose `input` field embeds the retrieved "Historical facts:[...]" the reader saw.
For each question it recomputes standard-protocol Hits@1 (the lenient
any-predicted-in-gold criterion that reproduces TimeR4's reported 72.8) and, from
the retrieved facts, the SELECTION HEADROOM: gold answer is present in the
retrieved context but TimeR4 chose wrong. That headroom is the ceiling a
reranking plug-in (our temporal discriminator) could recover WITHOUT improving
retrieval. Stratified by qtype/qlabel.

No GPU. Join is by line order (predictions.jsonl is aligned 1:1 with test.json).

Usage:
  PYTHONPATH=. python scripts/timer4_oracle_probe.py \
    --preds <extracted>/results/MultiTQ/finetuned_llama2/predictions.jsonl \
    --test  data/multitq/questions/test.json
"""
import argparse, ast, json, re
from collections import defaultdict


def parse_ans(s):
    if isinstance(s, list):
        s = s[0] if s else ""
    try:
        v = ast.literal_eval(s)
        return [v] if isinstance(v, str) else list(v)
    except Exception:
        return [str(s)]


def norm(x):
    return str(x).strip().lower()


def get_facts(inp):
    m = re.search(r"Historical facts:\s*(\[.*?\])\s*\nQuestion", inp, re.S)
    if not m:
        return []
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return []


def wb_present(g, blob):
    """Word-boundary phrase match (stricter than substring; avoids short-name inflation)."""
    return re.search(r"(?<![a-z0-9])" + re.escape(g) + r"(?![a-z0-9])", blob) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--test", default="data/multitq/questions/test.json")
    args = ap.parse_args()

    test = json.load(open(args.test))
    preds = [json.loads(l) for l in open(args.preds)]
    assert len(test) == len(preds), f"len mismatch {len(test)} vs {len(preds)}"
    mis = sum(1 for i in range(len(test)) if test[i]["question"] != preds[i]["question"])
    print(f"n={len(test)}  order-mismatches={mis}")

    cells = defaultdict(lambda: defaultdict(int))
    for i in range(len(test)):
        t, p = test[i], preds[i]
        gold = [norm(a) for a in t["answers"]]
        pr = {norm(a) for a in parse_ans(p["prediction"])}
        h = 1 if (pr & set(gold)) else 0
        blob = " ||| ".join(get_facts(p["input"])).lower()
        rec = any(wb_present(g, blob) for g in gold)
        for k in ("OVERALL", f"qtype:{t['qtype']}", f"qlabel:{t['qlabel']}",
                  f"ans:{t['answer_type']}"):
            c = cells[k]
            c["n"] += 1; c["hit"] += h; c["rec"] += rec
            if rec and not h:
                c["recoverable"] += 1

    def row(k):
        c = cells[k]; n = c["n"]
        if not n:
            return None
        return (f"{k:20s} n={n:6d} | hit={100*c['hit']/n:5.1f}% "
                f"recall={100*c['rec']/n:5.1f}% | selection_headroom="
                f"{100*c['recoverable']/n:5.1f}%  ceiling="
                f"{100*(c['hit']+c['recoverable'])/n:5.1f}%")

    order = ["OVERALL",
             "qtype:after_first", "qtype:before_last", "qtype:equal_multi",
             "qtype:before_after", "qtype:equal", "qtype:first_last",
             "qlabel:Single", "qlabel:Multiple", "ans:entity", "ans:time"]
    for k in order:
        r = row(k)
        if r:
            print(r)

    hd = 100 * cells["OVERALL"]["recoverable"] / cells["OVERALL"]["n"]
    print(f"\nGATE: overall selection headroom = {hd:.1f}pp  "
          f"-> {'GO' if hd >= 3 else 'KILL (retrieval-bound; disc cannot help via rerank)'}")


if __name__ == "__main__":
    main()
