"""CPU fusion sweep over a per-instance score dump (phase2_system_hybrid --dump_scores).

Once the GPU run has dumped (lp, disc, rr, valid, meta) per instance, ALL fusion
experiments become instant CPU arithmetic: score = lp + alpha*disc + beta*rr, with
optional gating of the reranker to entity-answer questions (it overrides the temporal
disc on time answers). Reports standard-protocol Hits@1 (Overall/Single/Multiple/
Entity/Time) for each (alpha, beta, gate). No GPU, no re-decode.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        d["lp"] = np.asarray(d["lp"]); d["disc"] = np.asarray(d["disc"])
        d["rr"] = np.asarray(d["rr"]) if d["rr"] is not None else None
        d["valid"] = np.asarray(d["valid"])
        rows.append(d)
    return rows


def rank_of_gold(fused, valid):
    for pos, idx in enumerate(np.argsort(-fused), 1):
        if valid[idx]:
            return pos
    return None


def score(rows, alpha, beta, gate):
    cells = defaultdict(lambda: [0, 0])   # key -> [hits, n]
    for d in rows:
        b = 0.0 if (gate == "entity" and d["answer_type"] == "time") else beta
        fused = d["lp"] + alpha * d["disc"] + (b * d["rr"] if d["rr"] is not None else 0.0)
        r = rank_of_gold(fused, d["valid"])
        h = int(r == 1)
        for k in ("overall", f"Q:{d['qlabel']}", f"A:{d['answer_type']}", f"T:{d['qtype']}"):
            cells[k][0] += h; cells[k][1] += 1
    return {k: 100 * v[0] / max(v[1], 1) for k, v in cells.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--betas", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2])
    ap.add_argument("--gates", nargs="+", default=["none", "entity"])
    args = ap.parse_args()
    rows = load(args.dump)
    print(f"loaded {len(rows)} instances | alpha={args.alpha}")
    print(f"{'gate':7s} {'beta':>5s} | {'Overall':>7s} {'Single':>7s} {'Multiple':>8s} "
          f"{'Entity':>7s} {'Time':>7s}")
    best = (None, -1)
    for gate in args.gates:
        for beta in args.betas:
            s = score(rows, args.alpha, beta, gate)
            print(f"{gate:7s} {beta:5g} | {s['overall']:7.1f} {s.get('Q:Single',0):7.1f} "
                  f"{s.get('Q:Multiple',0):8.1f} {s.get('A:entity',0):7.1f} {s.get('A:time',0):7.1f}")
            if s["overall"] > best[1]:
                best = ((gate, beta), s["overall"])
    print(f"\nbest: gate={best[0][0]} beta={best[0][1]:g} -> Overall {best[1]:.1f}")


if __name__ == "__main__":
    main()
