"""Build a MultiTQ KG index for the TVR checker and path conversion.

Two things Phase 0 needs from the raw ICEWS quadruples (`full.txt`):

1. **Candidate-fact lookup** for first/last ordering checks.  A first_last
   question ("in which year did X *last* request Y?") is judged by comparing the
   answer fact's timestamp against *all* KG facts matching the question
   signature.  We index:
     * ``sr2ots``  : (s, r)      -> sorted list of (timestamp, o)
     * ``sro2ts``  : (s, r, o)   -> sorted list of timestamps
     * ``or2sts``  : (o, r)      -> sorted list of (timestamp, s)   [object-anchored]

2. **Timestamp-augmented path rendering** validated on real facts
   (checker.path_format.temporal_path_to_string).

Output: ``data/multitq/index/kg_index.pkl`` (+ stats printed).
"""
import argparse
import os
import pickle
from bisect import insort
from collections import defaultdict

from checker.path_format import temporal_path_to_string, wrap_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg")
OUT_DIR = os.path.join(ROOT, "data", "multitq", "index")


def load_quads(path):
    quads = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            s, r, o, ts = parts
            quads.append((s, r, o, ts))
    return quads


def build_index(quads):
    sr2ots = defaultdict(list)   # (s, r) -> [(ts, o)]
    sro2ts = defaultdict(list)   # (s, r, o) -> [ts]
    or2sts = defaultdict(list)   # (o, r) -> [(ts, s)]
    entities, relations, timestamps = set(), set(), set()
    for s, r, o, ts in quads:
        insort(sr2ots[(s, r)], (ts, o))
        insort(sro2ts[(s, r, o)], ts)
        insort(or2sts[(o, r)], (ts, s))
        entities.add(s); entities.add(o); relations.add(r); timestamps.add(ts)
    return {
        "sr2ots": dict(sr2ots),
        "sro2ts": dict(sro2ts),
        "or2sts": dict(or2sts),
        "n_entities": len(entities),
        "n_relations": len(relations),
        "n_timestamps": len(timestamps),
        "n_facts": len(quads),
    }


def demo_paths(quads, index):
    """Render a few real 1-hop and 2-hop paths in the temporal format."""
    out = []
    # 1-hop: first fact.
    s, r, o, ts = quads[0]
    out.append(wrap_path(temporal_path_to_string([(s, r, o, ts)])))
    # 2-hop: extend from o if it has any outgoing edge.
    for (s2, r2), ots in index["sr2ots"].items():
        if s2 == o and ots:
            ts2, o2 = ots[0]
            out.append(wrap_path(temporal_path_to_string([(s, r, o, ts), (o, r2, o2, ts2)])))
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_file", default=os.path.join(KG_DIR, "full.txt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "kg_index.pkl"))
    args = ap.parse_args()

    print(f"loading quads from {args.kg_file} ...")
    quads = load_quads(args.kg_file)
    print(f"  {len(quads)} facts")
    index = build_index(quads)
    print(f"  entities={index['n_entities']} relations={index['n_relations']} "
          f"timestamps={index['n_timestamps']}")
    print(f"  keys: sr2ots={len(index['sr2ots'])} sro2ts={len(index['sro2ts'])} "
          f"or2sts={len(index['or2sts'])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved index -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    print("\n=== timestamp-augmented path rendering (real facts) ===")
    for p in demo_paths(quads, index):
        print("  " + p)


if __name__ == "__main__":
    main()
