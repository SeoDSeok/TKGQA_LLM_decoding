"""E1 full-test harness — resolve every MultiTQ test question to an evaluable
signature across all 6 qtypes, reusing the linker (topic/answer), the checker
qtype->operator map, and the KG index. Produces, per question:

  {quid, qtype, op_family, answer_type, time_level, qlabel, topic, relation,
   anchor, candidates:[(o,t)], gold_answers, resolvable}

`resolvable` = a topic linked + a relation matched + a candidate set built +
(for anchored ops) an anchor resolved. This is the CPU foundation for the
system-level GCR-vanilla vs +discriminator decoding comparison; coverage is
reported honestly per qtype (event anchors are the hard tail).
"""
import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from checker.timepoint import TimePoint
from checker.constraints import resolve_first_last, resolve_before_after, MULTITQ_QTYPE_MAP
from scripts.multitq_linker import MultiTQLinker, norm
from scripts.build_before_after_signatures import parse_anchor, strict_norm

KG = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg")
QDIR = os.path.join(ROOT, "data", "multitq", "questions")
IDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")

# genuine question scaffolding only — do NOT drop relation verbs (express, wish,
# want, make, meet, visit, sign, ...), which are exactly what matches KG relations.
_STOP = {"with", "that", "this", "from", "into", "about", "their", "which", "when",
         "what", "year", "time", "before", "after", "first", "last", "same", "month",
         "day", "country", "who", "whom", "did", "does", "was", "were", "the", "and",
         "for", "has", "have", "had", "been", "whose", "there", "then", "than"}


def rel_keywords(s):
    return {w for w in norm(s).split() if len(w) > 2 and w not in _STOP}


def stems(words):
    """4-char stems to bridge inflection (visit/visited, meet/meeting/meetings,
    negotiate/negotiation, sign/signed, appeal/appealed)."""
    return {w[:4] for w in words if len(w) >= 4}


def ent_qualifier_tokens(e):
    """Parenthetical qualifier tokens that norm() discards:
    Ministry_(Taiwan) -> {taiwan}, Head_of_Government_(Afghanistan) -> {afghanistan}.
    These are exactly what disambiguates entities sharing a normalized base name."""
    toks = set()
    for group in re.findall(r"\(([^)]*)\)", e):
        s = re.sub(r"[^a-z0-9 ]", " ", group.lower())
        toks |= {w for w in s.split() if len(w) > 2}
    return toks


class FullResolver:
    def __init__(self):
        self.linker = MultiTQLinker()
        idx = pickle.load(open(IDX, "rb"))
        self.sr2ots = idx["sr2ots"]
        self.sro2ts = idx["sro2ts"]
        self.or2sts = idx["or2sts"]       # (o,r) -> [(s,t)] : topic as OBJECT
        # relations available per entity, in each direction
        self.subj_rels = defaultdict(list)   # entity as subject
        self.obj_rels = defaultdict(list)    # entity as object
        for (s, r) in self.sr2ots:
            self.subj_rels[s].append(r)
        for (o, r) in self.or2sts:
            self.obj_rels[o].append(r)
        self.rel_kw = {r: rel_keywords(r) for r in self.linker.relations}
        self.rel_stem = {r: stems(kw) for r, kw in self.rel_kw.items()}

    def best_relation(self, e, qwords):
        """Best (relation, direction, score): topic as subject or as object.
        Score = exact keyword overlap + 0.5*stem overlap (bridges inflection)."""
        qstem = stems(qwords)
        best = (None, None, 0.0)
        for rels, d in ((self.subj_rels.get(e, ()), "subj"), (self.obj_rels.get(e, ()), "obj")):
            for r in rels:
                sc = len(self.rel_kw.get(r, set()) & qwords) + 0.5 * len(self.rel_stem.get(r, set()) & qstem)
                if sc > best[2]:
                    best = (r, d, sc)
        return best

    def event_anchor_time(self, e, r, direction, anchor_ents):
        """Event anchor time: the time linking an anchor entity to the topic via r,
        tried in both directions. e is the topic; direction tells which slot e is."""
        for ae in anchor_ents:
            # topic=subject: (topic, r, anchor) ; or anchor is a co-subject to same object
            for key in ((e, r, ae), (ae, r, e)):
                ts = self.sro2ts.get(key)
                if ts:
                    return TimePoint.parse(sorted(ts)[0])
            # anchor as a subject reaching the topic-as-object via r
            for (s2, t2) in self.or2sts.get((e, r), ()):
                if s2 == ae:
                    return TimePoint.parse(t2)
            # anchor as an object of the topic-as-subject via r
            for (t2, o2) in self.sr2ots.get((e, r), ()):
                if o2 == ae:
                    return TimePoint.parse(t2)
        return None

    def resolve(self, q):
        qtype = q["qtype"]; qwords = {w for w in norm(q["question"]).split() if len(w) > 2}
        topics = self.linker.link_entities(q["question"])
        out = {"quid": q["quid"], "qtype": qtype, "answer_type": q["answer_type"],
               "time_level": q["time_level"], "qlabel": q["qlabel"],
               "resolvable": False, "topic": None, "relation": None,
               "anchor": None, "candidates": [], "gold_answers": q["answers"]}
        if not topics:
            return out
        # choose (topic, relation, direction). Disambiguate entities that share a
        # normalized name (Ministry_(Taiwan) vs Ministry_(Afghanistan)) by base-name
        # token overlap PLUS a strongly-weighted parenthetical-qualifier match --
        # norm() strips "(Taiwan)", so the qualifier is the only real discriminator
        # between two same-named entities.
        qtok = set(norm(q["question"]).split())
        def ent_overlap(x):
            base = len({w for w in norm(x).split() if len(w) > 2} & qtok)
            qual = len(ent_qualifier_tokens(x) & qtok)
            return base + 2.0 * qual
        best = (None, None, None, -1.0)
        for e in topics:
            r, d, sc = self.best_relation(e, qwords)
            if not r or sc <= 0:      # require a real relation match
                continue
            total = sc + 0.6 * ent_overlap(e)
            if total > best[3]:
                best = (e, r, d, total)
        e, r, direction, sc = best
        if e is None:
            return out
        out["topic"], out["relation"], out["direction"] = e, r, direction
        if direction == "subj":       # topic is subject; candidates = objects
            cands = [(o, t) for (t, o) in self.sr2ots.get((e, r), [])]
        else:                         # topic is object; candidates = subjects
            cands = [(s, t) for (t, s) in self.or2sts.get((e, r), [])]
        if len(cands) < 1:
            return out
        # first_last only: if the question also names the OTHER argument (a second
        # linked entity among the candidates), constrain to it -- "when did s r <o>"
        # uses (s,r,o) timestamps, not (s,r,*). (For anchored ops the second entity
        # is the ANCHOR, handled below, so we must not filter candidates by it.)
        if qtype == "first_last":
            cand_ents = {o for o, _ in cands}
            others = [x for x in topics if x != e and x in cand_ents]
            if others:
                cands = [(o, t) for (o, t) in cands if o == others[0]]
                out["constrained_to"] = others[0]
        out["candidates"] = cands
        s = e  # subject alias for anchor lookup below (approx; direction-aware anchor next)
        fam = MULTITQ_QTYPE_MAP[qtype]

        # anchor resolution for anchored operators
        needs_anchor = qtype in ("before_after", "equal", "equal_multi", "after_first", "before_last")
        anchor = None
        if needs_anchor:
            anchor = parse_anchor(q["question"])          # explicit date
            if anchor is None:                            # event anchor (entity)
                anchor_ents = [x for x in topics if x != e]
                anchor = self.event_anchor_time(e, r, direction, anchor_ents)
            if anchor is None:
                return out                                # cannot resolve -> unresolvable
            out["anchor"] = str(anchor)
        out["op_family"] = fam["primary"]
        out["resolvable"] = True
        # apply the gold operator to the candidate set -> predicted answer(s);
        # answer-recall = predicted intersects the gold answer set (the ceiling
        # that GCR-vanilla / +disc decoding can reach on this candidate set).
        out["pred"] = self.apply_operator(qtype, q["question"], cands, anchor,
                                          q["answer_type"], q["time_level"])
        gold = {strict_norm(a) for a in q["answers"]}
        pred_norm = {strict_norm(p) for p in out["pred"]}
        out["answer_hit"] = len(pred_norm & gold) > 0
        return out

    def apply_operator(self, qtype, question, cands, anchor, answer_type, time_level):
        """Return predicted answer(s) (entity strings or time strings)."""
        tps = [(o, TimePoint.parse(t)) for (o, t) in cands]
        if qtype in ("equal", "equal_multi"):
            if anchor is None:
                return []
            hit = [(o, tp) for (o, tp) in tps if tp.equal_at(anchor)]
            return _ans(hit, answer_type, time_level)
        if qtype == "first_last":
            if not tps:
                return []
            key = min if resolve_first_last(question) == "first" else max
            return _ans([key(tps, key=lambda x: x[1])], answer_type, time_level)
        if qtype == "before_after":
            if anchor is None:
                return []
            ba = resolve_before_after(question)
            surv = [(o, tp) for (o, tp) in tps
                    if (tp.strictly_before(anchor) if ba == "before" else tp.strictly_after(anchor))]
            return _ans(surv, answer_type, time_level)
        if qtype in ("after_first", "before_last"):
            if anchor is None:
                return []
            fam = MULTITQ_QTYPE_MAP[qtype]
            surv = [(o, tp) for (o, tp) in tps
                    if (tp.strictly_after(anchor) if fam["interval_op"] == "after"
                        else tp.strictly_before(anchor))]
            if not surv:
                return []
            key = min if fam["ordering"] == "first" else max
            return _ans([key(surv, key=lambda x: x[1])], answer_type, time_level)
        return []


def _fmt_time(tp, level):
    if level == "year":
        return f"{tp.year:04d}"
    if level == "month":
        return f"{tp.year:04d}-{tp.month:02d}"
    return f"{tp.year:04d}-{tp.month:02d}-{tp.day:02d}"


def _ans(pairs, answer_type, time_level):
    """pairs = [(entity, TimePoint)]. Format per answer_type / question granularity."""
    if answer_type == "time":
        return list({_fmt_time(tp, time_level) for _, tp in pairs})
    return [o for o, _ in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("-n", type=int, default=3000, help="sample size (0=all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    R = FullResolver()
    qs = json.load(open(f"{QDIR}/{args.split}.json"))
    if args.n:
        # stratified sample across qtypes
        by = defaultdict(list)
        for q in qs:
            by[q["qtype"]].append(q)
        per = max(1, args.n // len(by))
        qs = [q for lst in by.values() for q in lst[:per]]
    cov = defaultdict(lambda: [0, 0])   # resolvable
    rec = defaultdict(lambda: [0, 0])   # oracle answer-recall among resolvable
    kdist = []
    sigs = []
    for q in qs:
        r = R.resolve(q)
        cov[q["qtype"]][1] += 1
        if r["resolvable"]:
            cov[q["qtype"]][0] += 1
            kdist.append(len(r["candidates"]))
            rec[q["qtype"]][1] += 1
            rec[q["qtype"]][0] += int(r.get("answer_hit", False))
        sigs.append(r)
    print(f"Full-test on {len(qs)} questions (stratified). "
          f"resolvable = topic+relation+candidates(+anchor); "
          f"oracle-recall = gold reachable by the operator on our candidates.")
    print(f"  {'qtype':14s} {'resolvable':>12s}  {'oracle-recall':>14s}")
    tot, rtot = [0, 0], [0, 0]
    for qt in ("equal", "equal_multi", "first_last", "before_after", "after_first", "before_last"):
        ok, n = cov[qt]; rok, rn = rec[qt]
        tot[0] += ok; tot[1] += n; rtot[0] += rok; rtot[1] += rn
        print(f"  {qt:14s} {100*ok/max(n,1):9.1f}%  {100*rok/max(rn,1):12.1f}%  ({rok}/{rn})")
    print(f"  {'OVERALL':14s} {100*tot[0]/max(tot[1],1):9.1f}%  {100*rtot[0]/max(rtot[1],1):12.1f}%")
    if kdist:
        import numpy as np
        print(f"  candidate-set size: mean {np.mean(kdist):.1f} median {int(np.median(kdist))}")
    if args.out:
        with open(args.out, "w") as f:
            for s in sigs:
                f.write(json.dumps(s) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
