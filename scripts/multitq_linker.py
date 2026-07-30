"""Entity/relation linking for MultiTQ questions (no gold annotations exist).

MultiTQ questions carry only NL text + answers, so to build GCR training data and
to measure real TVR we must recover each question's KG signature. Naive substring
matching is ~100% recall but terrible precision ("ministry" hits dozens of
`Ministry_(*)` entities). We use:

  1. boundary-aware, longest-match entity name detection (drop a match subsumed
     by a longer match covering the same span);
  2. exact answer linking (answers are KG canonical labels up to normalization);
  3. a **KG connectivity filter**: keep a (topic, answer) pair only if topic
     reaches answer within K hops in the (time-blind) TKG. Connectivity is the
     disambiguator that kills false-positive topics.

Reports honest coverage: fraction of questions with a connectivity-supported
(topic, answer) link (entity answers) or a linked topic (time answers).
"""
import argparse
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg")
QDIR = os.path.join(ROOT, "data", "multitq", "questions")


def norm(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r"\s*\([^)]*\)", "", s)   # drop parentheticals
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


class MultiTQLinker:
    def __init__(self):
        self.entities = list(json.load(open(f"{KG}/entity2id.json")).keys())
        self.relations = list(json.load(open(f"{KG}/relation2id.json")).keys())
        # normalized name -> list of canonical entities
        self.ent_by_norm = defaultdict(list)
        for e in self.entities:
            self.ent_by_norm[norm(e)].append(e)
        # candidate names sorted longest-first for greedy longest match
        self._names = sorted((n for n in self.ent_by_norm if len(n) >= 3),
                             key=len, reverse=True)
        # adjacency (time-blind) for connectivity
        self.adj = defaultdict(set)
        self._load_adjacency()

    def _load_adjacency(self):
        with open(f"{KG}/full.txt") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 4:
                    s, r, o, t = p
                    self.adj[s].add(o)
                    self.adj[o].add(s)  # undirected for reachability

    def link_entities(self, question: str) -> list[str]:
        """Boundary-aware longest-match entity linking."""
        q = " " + norm(question) + " "
        used_spans = []
        found = []
        for name in self._names:
            pat = " " + name + " "
            idx = q.find(pat)
            if idx == -1:
                continue
            span = (idx, idx + len(pat))
            # skip if overlapped by an already-accepted (longer) match
            if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
                continue
            used_spans.append(span)
            found.extend(self.ent_by_norm[name])
        return found

    def link_answers(self, answers: list[str]) -> set[str]:
        out = set()
        for a in answers:
            out |= set(self.ent_by_norm.get(norm(a), []))
        return out

    def reachable(self, src: str, dst: str, k: int = 2) -> bool:
        if src == dst:
            return True
        frontier = {src}
        seen = {src}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                for v in self.adj.get(u, ()):  # noqa
                    if v == dst:
                        return True
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
        return False

    def link(self, q: dict, k: int = 2) -> dict:
        topics = self.link_entities(q["question"])
        result = {"topics": topics, "answer_entities": [], "supported_pairs": [],
                  "answer_type": q["answer_type"]}
        if q["answer_type"] == "time":
            # time answers: a linked topic is enough to anchor the signature
            result["ok"] = len(topics) > 0
            return result
        ans = self.link_answers(q["answers"])
        result["answer_entities"] = list(ans)
        pairs = []
        for t in topics:
            for a in ans:
                if self.reachable(t, a, k):
                    pairs.append((t, a))
        result["supported_pairs"] = pairs
        result["ok"] = len(pairs) > 0
        return result


def measure(n=2000, k=2):
    linker = MultiTQLinker()
    qs = json.load(open(f"{QDIR}/test.json"))[:n]
    from collections import Counter
    by_type = defaultdict(lambda: [0, 0])   # qtype -> [ok, total]
    ans_type = defaultdict(lambda: [0, 0])
    n_topics = []
    ok_total = 0
    for q in qs:
        r = linker.link(q, k=k)
        by_type[q["qtype"]][1] += 1
        ans_type[q["answer_type"]][1] += 1
        n_topics.append(len(r["topics"]))
        if r["ok"]:
            ok_total += 1
            by_type[q["qtype"]][0] += 1
            ans_type[q["answer_type"]][0] += 1
    print(f"MultiTQ linking coverage on {len(qs)} test questions (k={k} hops):")
    print(f"  overall usable: {100*ok_total/len(qs):.1f}%")
    print(f"  mean #topic candidates/question: {sum(n_topics)/len(n_topics):.2f}")
    print("  by answer_type:")
    for at, (ok, tot) in ans_type.items():
        print(f"    {at:6s} {100*ok/tot:5.1f}%  ({ok}/{tot})")
    print("  by qtype:")
    for qt, (ok, tot) in sorted(by_type.items()):
        print(f"    {qt:14s} {100*ok/tot:5.1f}%  ({ok}/{tot})")
    return {
        "n": len(qs), "k": k, "overall": ok_total / len(qs),
        "mean_topics": sum(n_topics) / len(n_topics),
        "by_answer_type": {at: v[0] / v[1] for at, v in ans_type.items()},
        "by_qtype": {qt: v[0] / v[1] for qt, v in by_type.items()},
        "by_qtype_counts": {qt: v for qt, v in by_type.items()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("-k", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "linking_coverage.json"))
    args = ap.parse_args()
    stats = measure(args.n, args.k)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(stats, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
