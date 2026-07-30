"""Model-free early signal for hypothesis H1 (Phase 0 pre-check).

H1: GCR's KG-Trie enforces only *structural* validity, so for temporal questions
the constrained candidate space contains many structurally-faithful but
temporally-invalid paths.

This script measures the *temporal pollution of the structural candidate space*
directly from the MultiTQ TKG index — NO model, NO training, NO entity linking.
It is an UPPER BOUND on the violation opportunity (how wrong the space is), not
the model's actual generation TVR (which needs the trained model). But if the
space is already highly polluted on complex operators, H1 is plausible and the
full pipeline is worth building; if it is near-zero, we stop early (the plan's
<5% early-abort criterion).

Method: MultiTQ questions are templated from KG signatures.
* first/last (answer_type=time)   -> signature (s, r, o); answer = min/max of that
  group's timestamps. A time-blind trie admits all k timestamps -> only 1 correct,
  so violation opportunity = (k-1)/k.
* first/last (answer_type=entity) -> signature (s, r); answer = object at min/max
  time among all (o, ts). Time-blind admits all distinct objects -> opportunity =
  (#distinct objects - 1) / #distinct objects.
* before/after -> signature group; for an anchor drawn from the group's own
  timestamps, the expected fraction on the wrong side is reported.

We weight by the actual MultiTQ test qtype mix so the headline number reflects
the benchmark.
"""
import json
import os
import pickle
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")
QTEST = os.path.join(ROOT, "data", "multitq", "questions", "test.json")


def load():
    with open(INDEX, "rb") as f:
        idx = pickle.load(f)
    qt = Counter(q["qtype"] for q in json.load(open(QTEST)))
    return idx, qt


def frac_stats(values):
    values = list(values)
    n = len(values)
    return {
        "groups": n,
        "mean": sum(values) / n if n else 0.0,
    }


def main():
    idx, qtest = load()
    sro2ts = idx["sro2ts"]      # (s,r,o) -> [ts]
    sr2ots = idx["sr2ots"]      # (s,r)   -> [(ts,o)]

    # --- first/last on (s,r,o): answer is one timestamp among the group ---
    time_opp = []          # (k-1)/k per group with k distinct timestamps
    multi_ts_groups = 0
    for key, ts_list in sro2ts.items():
        k = len(set(ts_list))
        if k >= 1:
            time_opp.append((k - 1) / k)
            if k >= 2:
                multi_ts_groups += 1
    time_stat = frac_stats(time_opp)
    pct_multi_sro = 100 * multi_ts_groups / len(sro2ts)

    # --- first/last on (s,r): answer is the object at the extreme time ---
    ent_opp = []
    multi_obj_groups = 0
    for key, ots in sr2ots.items():
        objs = set(o for _, o in ots)
        m = len(objs)
        if m >= 1:
            ent_opp.append((m - 1) / m)
            if m >= 2:
                multi_obj_groups += 1
    ent_stat = frac_stats(ent_opp)
    pct_multi_sr = 100 * multi_obj_groups / len(sr2ots)

    # --- before/after on (s,r): anchor = median timestamp of the group;
    #     opportunity = fraction of candidate objects whose time is on the
    #     "wrong" side relative to a random target side. For a balanced split,
    #     ~half the structurally-valid candidates violate a given before/after. ---
    ba_opp = []
    for key, ots in sr2ots.items():
        times = sorted(t for t, _ in ots)
        k = len(times)
        if k >= 2:
            mid = times[k // 2]
            before = sum(1 for t in times if t < mid)
            after = sum(1 for t in times if t > mid)
            # a before(mid) question: candidates that are >= mid violate
            frac_violate_before = (k - before) / k
            ba_opp.append(frac_violate_before)
    ba_stat = frac_stats(ba_opp)

    # --- headline: weight operator opportunities by MultiTQ test qtype mix ---
    # map each qtype to its dominant structural opportunity source
    total = sum(qtest.values())
    # equal: a time-blind trie admits all timestamps of the signature; only the
    # matching-granularity ones are valid -> reuse (k-1)/k on (s,r,o) as proxy.
    weights = {
        "equal": time_stat["mean"],
        "equal_multi": time_stat["mean"],
        "before_after": ba_stat["mean"],
        "first_last": (time_stat["mean"] + ent_stat["mean"]) / 2,
        "after_first": (time_stat["mean"] + ent_stat["mean"]) / 2,
        "before_last": (time_stat["mean"] + ent_stat["mean"]) / 2,
    }
    headline = sum(qtest[q] * weights[q] for q in qtest) / total

    print("=" * 68)
    print("MultiTQ structural temporal-pollution probe (model-free, H1 pre-check)")
    print("=" * 68)
    print(f"KG facts indexed: {idx['n_facts']}  |  entities {idx['n_entities']}  "
          f"relations {idx['n_relations']}  timestamps {idx['n_timestamps']}")
    print()
    print(f"(s,r,o) groups: {len(sro2ts)}")
    print(f"  with >=2 distinct timestamps: {pct_multi_sro:.1f}%  "
          f"(first/last-time 'trap' groups)")
    print(f"  mean time-blind violation opportunity (k-1)/k: {time_stat['mean']:.3f}")
    print()
    print(f"(s,r) groups: {len(sr2ots)}")
    print(f"  with >=2 distinct objects: {pct_multi_sr:.1f}%")
    print(f"  mean object opportunity (m-1)/m: {ent_stat['mean']:.3f}")
    print(f"  mean before/after wrong-side fraction: {ba_stat['mean']:.3f}")
    print()
    print("Per-operator structural violation opportunity (weighted by test mix):")
    for q in sorted(qtest, key=lambda x: -qtest[x]):
        print(f"  {q:14s} n={qtest[q]:6d}  opportunity={weights[q]:.3f}")
    print("-" * 68)
    print(f"HEADLINE structural violation opportunity (test-weighted): "
          f"{headline:.3f}  ({100*headline:.1f}%)")
    print("-" * 68)
    print("NOTE: this is the pollution of the *structural candidate space* — an")
    print("upper bound on violation opportunity, NOT the trained model's TVR.")
    print("A high value means H1 is plausible; the real number needs GCR-vanilla")
    print("generation on a MultiTQ-trained model (next GPU session).")

    out = os.path.join(ROOT, "results", "h1_structural_pollution.md")
    with open(out, "w") as f:
        f.write("# H1 pre-check: MultiTQ structural temporal pollution (model-free)\n\n")
        f.write(f"- KG: {idx['n_facts']} facts, {idx['n_timestamps']} timestamps\n")
        f.write(f"- (s,r,o) groups with >=2 timestamps: **{pct_multi_sro:.1f}%**\n")
        f.write(f"- (s,r) groups with >=2 objects: **{pct_multi_sr:.1f}%**\n")
        f.write(f"- mean first/last time opportunity (k-1)/k: **{time_stat['mean']:.3f}**\n")
        f.write(f"- mean first/last entity opportunity (m-1)/m: **{ent_stat['mean']:.3f}**\n")
        f.write(f"- mean before/after wrong-side fraction: **{ba_stat['mean']:.3f}**\n\n")
        f.write("| qtype | test n | structural violation opportunity |\n|---|--:|--:|\n")
        for q in sorted(qtest, key=lambda x: -qtest[x]):
            f.write(f"| {q} | {qtest[q]} | {weights[q]:.3f} |\n")
        f.write(f"\n**Headline (test-weighted): {100*headline:.1f}%**\n\n")
        f.write("> Upper bound on violation *opportunity* (space pollution), NOT the "
                "trained model's generation TVR. High value ⇒ H1 plausible ⇒ build "
                "the full pipeline; the definitive number needs a MultiTQ-trained "
                "GCR-vanilla model.\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
