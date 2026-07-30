"""Candidate-set dataset for the temporal discriminator v0.

Each training/eval item is a *question-conditioned candidate set* — exactly the
isolated Phase-0 setting where (s, r) are fixed and only the timestamp choice is
in question:

  first/last : candidates = every timestamp for the (s, r, o) fact; the operator
               makes the earliest (first) or latest (last) the single gold.
  before/after: candidates = every (o_i, t_i) fact for (s, r) (resolved from the
               KG index `sr2ots`); valid iff t_i is strictly before / after the
               question anchor (granularity-aware, via the checker's TimePoint).

Features are deliberately **leakage-safe** (design §2.1 / §6.5): a candidate is
described only by value-space time (Φ inputs t_rel, t_abs), plus frozen text
embeddings of its object/relation strings. No chronological-rank, is_min or
is_max flag is exposed — the model must derive argmin/argmax from Φ via set
attention, which is the whole hypothesis under test.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import defaultdict

import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)
from checker.timepoint import TimePoint
from discriminator.encode_time import date_to_fracyear

SIG = os.path.join(ROOT, "data", "multitq", "signatures")
CACHE = os.path.join(ROOT, "data", "multitq", "discriminator_v0")
os.makedirs(CACHE, exist_ok=True)

ABS_CENTER, ABS_SCALE = 2010.0, 10.0
EPS = 1e-6


# --------------------------------------------------------------------------- #
# 1. build raw candidate-set examples                                         #
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return s.replace("_", " ").strip()


def build_examples(split: str, ops=("first", "last", "before", "after"),
                   k_max_before_after: int = 40):
    """Return a list of candidate-set dicts for the requested operators."""
    ex = []
    if any(o in ops for o in ("first", "last")):
        ex += _build_first_last(split, {o for o in ops if o in ("first", "last")})
    if any(o in ops for o in ("before", "after")):
        ex += _build_before_after(split, {o for o in ops if o in ("before", "after")},
                                  k_max_before_after)
    return ex


def _build_first_last(split, ops):
    path = os.path.join(SIG, f"first_last_time_{split}.jsonl")
    out = []
    for line in open(path):
        d = json.loads(line)
        if d["op"] not in ops or d["n_candidates"] < 2:
            continue
        tps = sorted({TimePoint.parse(t) for t in d["candidate_timestamps"]})
        gold = tps[0] if d["op"] == "first" else tps[-1]
        cands = [{"o": d["o"], "r": d["r"], "frac": date_to_fracyear(tp),
                  "t_str": str(tp), "valid": int(tp == gold)} for tp in tps]
        out.append({"quid": d["quid"], "op": d["op"], "question": d["question"],
                    "s": d["s"], "cands": cands, "anchor_frac": None})
    return out


def _load_kg():
    idx = pickle.load(open(os.path.join(ROOT, "data/multitq/index/kg_index.pkl"), "rb"))
    return idx["sr2ots"]  # (s, r) -> [(t_str, o_str), ...]


def _load_s2facts():
    sr2ots = _load_kg()
    s2 = defaultdict(list)
    for (s, r), ots in sr2ots.items():
        for t, o in ots:
            s2[s].append((r, o, t))
    return s2


def build_during_examples(split, per=4000, seed=0, k_max=16):
    """Synthetic `during` (interval-containment) sets from KG (s,r) timestamp pools.
    Candidates = (o,t) facts for (s,r); a random interval [lo,hi] is drawn from the
    pool's own dates; valid = lo <= t <= hi. Gives the interval scorer real
    two-sided-bound supervision (the operator absent from MultiTQ)."""
    import numpy as _np
    rng = _np.random.RandomState(seed)
    sr2ots = _load_kg()
    seen, out = set(), []
    path = os.path.join(SIG, f"before_after_{split}.jsonl")
    for line in open(path):
        d = json.loads(line)
        key = (d["s"], d["r"])
        if key in seen:
            continue
        seen.add(key)
        facts = sr2ots.get(key)
        if not facts or len({t for t, _ in facts}) < 3:
            continue
        tps = sorted({TimePoint.parse(t) for t, _ in facts})
        if len(tps) < 3:
            continue
        i, j = sorted(rng.choice(len(tps), 2, replace=False))
        if j - i < 1 or (i == 0 and j == len(tps) - 1):
            continue
        lo_tp, hi_tp = tps[i], tps[j]
        cands = []
        for t, o in facts:
            tp = TimePoint.parse(t)
            inside = (not tp.strictly_before(lo_tp)) and (not tp.strictly_after(hi_tp))
            cands.append({"o": o, "r": d["r"], "frac": date_to_fracyear(tp),
                          "t_str": str(tp), "valid": int(inside)})
        nv = sum(c["valid"] for c in cands)
        if nv == 0 or nv == len(cands) or len(cands) < 2:
            continue
        if len(cands) > k_max:
            v = [c for c in cands if c["valid"]]; iv = [c for c in cands if not c["valid"]]
            rng.shuffle(iv); cands = (v + iv)[:k_max]
            if not any(c["valid"] for c in cands):
                cands[0] = v[0]
        q = f"What involving {_norm(d['s'])} happened between {lo_tp} and {hi_tp}?"
        out.append({"quid": 90_000_000 + len(out), "op": "during", "question": q,
                    "s": d["s"], "cands": cands,
                    "anchor_frac": date_to_fracyear(lo_tp),
                    "anchor2_frac": date_to_fracyear(hi_tp)})
        if len(out) >= per:
            break
    return out


def build_khop_examples(split, k_max=40, per_op=None, seed=0):
    """T3 full-subgraph candidate sets: candidates = ALL facts of the subject s
    (every relation/object), gold = the answer fact `(r, o, extremal_t)`. Forces
    the scorer to match relation/object from the question AND pick the timestamp."""
    import numpy as _np
    rng = _np.random.RandomState(seed)
    s2 = _load_s2facts()
    out = defaultdict(list)
    for line in open(os.path.join(SIG, f"first_last_time_{split}.jsonl")):
        d = json.loads(line)
        if d["n_candidates"] < 2:
            continue
        op = d["op"]
        if per_op and len(out[op]) >= per_op:
            continue
        tps = sorted({TimePoint.parse(t) for t in d["candidate_timestamps"]})
        gold_t = str(tps[0] if op == "first" else tps[-1])
        gold = (d["r"], d["o"], gold_t)
        allf = s2.get(d["s"], []) + [gold]
        # ALWAYS include the full sibling timestamp group (same r,o, other t) so the
        # temporal decision is present; fill the rest with other-relation distractors.
        sibs = [f for f in allf if f[0] == d["r"] and f[1] == d["o"]]
        others = list({f for f in allf if not (f[0] == d["r"] and f[1] == d["o"])})
        rng.shuffle(others)
        keep = list({f for f in sibs}) + others[: max(0, k_max - len(set(sibs)))]
        keep = list({f for f in keep})
        if gold not in keep:
            keep.append(gold)
        rng.shuffle(keep)
        cands = [{"r": r, "o": o, "t_str": t, "frac": date_to_fracyear(TimePoint.parse(t)),
                  "valid": int((r, o, t) == gold)} for (r, o, t) in keep]
        out[op].append({"quid": d["quid"], "op": op, "question": d["question"],
                        "s": d["s"], "cands": cands, "anchor_frac": None, "gold": gold})
    return [e for op in out for e in out[op]]


def _build_before_after(split, ops, k_max):
    path = os.path.join(SIG, f"before_after_{split}.jsonl")
    sr2ots = _load_kg()
    out = []
    for line in open(path):
        d = json.loads(line)
        if d["op"] not in ops:
            continue
        facts = sr2ots.get((d["s"], d["r"]))
        if not facts:
            continue
        anchor = TimePoint.parse(d["anchor"], d.get("granularity"))
        cands = []
        for t_str, o in facts:
            tp = TimePoint.parse(t_str)
            if d["op"] == "before":
                valid = tp.strictly_before(anchor)
            else:
                valid = tp.strictly_after(anchor)
            cands.append({"o": o, "r": d["r"], "frac": date_to_fracyear(tp),
                          "t_str": str(tp), "valid": int(valid)})
        # need at least one valid and one invalid for a meaningful set
        nv = sum(c["valid"] for c in cands)
        if nv == 0 or nv == len(cands) or len(cands) < 2:
            continue
        if len(cands) > k_max:  # keep all valids + fill with invalids (train only)
            valids = [c for c in cands if c["valid"]]
            invalids = [c for c in cands if not c["valid"]]
            keep = valids + invalids[: max(0, k_max - len(valids))]
            cands = keep[:k_max]
        out.append({"quid": d["quid"], "op": d["op"], "question": d["question"],
                    "s": d["s"], "cands": cands, "anchor_frac": date_to_fracyear(anchor),
                    "answers": d.get("answers", [])})
    return out


# --------------------------------------------------------------------------- #
# 2. frozen text embeddings (MiniLM) — questions, objects, relations          #
# --------------------------------------------------------------------------- #
def _cache_key(strings):
    h = hashlib.md5("".join(sorted(strings)).encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"textemb_{h}.pt")


def build_text_embeddings(examples, device="cuda", batch=512):
    """Frozen all-MiniLM-L6-v2 embeddings for every question / object / relation."""
    strings = set()
    for e in examples:
        strings.add(("q", e["question"]))
        for c in e["cands"]:
            strings.add(("e", c["o"]))
            strings.add(("e", c["r"]))
    keyset = [f"{t}:{s}" for t, s in strings]
    cache = _cache_key(keyset)
    if os.path.exists(cache):
        return torch.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    items = list(strings)
    texts = [(_norm(s) if t == "e" else s) for t, s in items]
    vecs = model.encode(texts, batch_size=batch, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    emb = {f"{t}:{s}": torch.tensor(v) for (t, s), v in zip(items, vecs)}
    torch.save(emb, cache)
    return emb


# --------------------------------------------------------------------------- #
# 3. torch Dataset + collate                                                  #
# --------------------------------------------------------------------------- #
# abstract operator polarity (wants_later, is_threshold) — the LLM supplies this
# zero-shot for held-out operators (see llm_polarity.py). Conditioning on this
# instead of the question text makes the scorer operator-agnostic.
OP_POLARITY = {"first": (0.0, 0.0), "last": (1.0, 0.0),
               "before": (0.0, 1.0), "after": (1.0, 1.0)}

# interval polarity: (dir, uses_lower, uses_upper). dir ∈ {-1 min, 0 none, +1 max}.
# Unifies every operator as an interval predicate (Allen-style) + a superlative
# direction: first/last = no bound + min/max; before = upper bound; after = lower
# bound; during/equal = both bounds. Anchors come from anchor_frac (lower) and
# anchor2_frac (upper); before uses anchor_frac as its UPPER bound.
OP_INTERVAL = {"first": (-1.0, 0, 0), "last": (1.0, 0, 0),
               "after": (0.0, 1, 0), "before": (0.0, 0, 1),
               "during": (0.0, 1, 1), "equal": (0.0, 1, 1),
               # composites = one bound + a superlative direction (zero-shot from the
               # bound/dir primitives, never trained as such); equal_multi ~ equal.
               "after_first": (-1.0, 1, 0), "before_last": (1.0, 0, 1),
               "equal_multi": (0.0, 1, 1)}


class CandidateSetDataset(Dataset):
    def __init__(self, examples, text_emb, use_rank_leak=False, polarity_override=None,
                 abs_only=False):
        self.ex = examples
        self.emb = text_emb
        self.use_rank_leak = use_rank_leak  # ablation only
        # ablation: drop the group-min/max rescaling of the relative channel, so the
        # model sees only absolute time and NO within-group position hint. Tests
        # whether superlatives come from value-space Φ or from the set-rescaling.
        self.abs_only = abs_only
        # optional {quid: op-label} overriding the polarity source (LLM zero-shot)
        self.polarity_override = polarity_override or {}
        self.edim = next(iter(text_emb.values())).shape[0]

    def __len__(self):
        return len(self.ex)

    def _q(self, s):
        return self.emb[f"q:{s}"]

    def _e(self, s):
        return self.emb[f"e:{s}"]

    def __getitem__(self, i):
        e = self.ex[i]
        fracs = torch.tensor([c["frac"] for c in e["cands"]], dtype=torch.float32)
        lo, hi = fracs.min().item(), fracs.max().item()
        span = max(hi - lo, EPS)
        t_abs = (fracs - ABS_CENTER) / ABS_SCALE
        # relative channel: group-rescaled by default; abs_only removes the hint.
        t_rel = t_abs.clone() if getattr(self, "abs_only", False) else (fracs - lo) / span
        o_emb = torch.stack([self._e(c["o"]) for c in e["cands"]])
        r_emb = torch.stack([self._e(c["r"]) for c in e["cands"]])
        valid = torch.tensor([c["valid"] for c in e["cands"]], dtype=torch.float32)
        # question + anchor channel
        q_emb = self._q(e["question"])
        if e["anchor_frac"] is None:
            a_rel, a_abs, has_a = 0.0, 0.0, 0.0
        else:
            a_rel = (e["anchor_frac"] - lo) / span
            a_abs = (e["anchor_frac"] - ABS_CENTER) / ABS_SCALE
            has_a = 1.0
        anchor = torch.tensor([has_a, a_rel, a_abs], dtype=torch.float32)
        # polarity: gold op, or an override (e.g. LLM zero-shot prediction)
        pol_op = self.polarity_override.get(e["quid"], e["op"])
        wl, thr = OP_POLARITY.get(pol_op, (0.0, 1.0))
        polarity = torch.tensor([wl, thr, has_a, a_rel, a_abs], dtype=torch.float32)
        # interval polarity [dir, hl, hu, L_rel, L_abs, U_rel, U_abs]
        dr, uses_l, uses_u = OP_INTERVAL.get(pol_op, (0.0, 0, 0))
        # upper-bound-only ops (before, before_last) route their single anchor to the
        # UPPER bound; everything else uses anchor_frac as the LOWER bound (+ anchor2 as
        # upper for two-bound ops). Driven by the bound flags so composites route right.
        a2 = e.get("anchor2_frac")
        if uses_u and not uses_l:
            L_f, U_f = None, e.get("anchor_frac")
        else:
            L_f, U_f = e.get("anchor_frac"), a2
        def _rel_abs(f):
            if f is None:
                return 0.0, 0.0
            return (f - lo) / span, (f - ABS_CENTER) / ABS_SCALE
        Lr, La = _rel_abs(L_f); Ur, Ua = _rel_abs(U_f)
        polarity_iv = torch.tensor([dr, float(uses_l), float(uses_u), Lr, La, Ur, Ua],
                                   dtype=torch.float32)
        item = {"t_rel": t_rel, "t_abs": t_abs, "o_emb": o_emb, "r_emb": r_emb,
                "valid": valid, "q_emb": q_emb, "anchor": anchor,
                "polarity": polarity, "polarity_iv": polarity_iv, "op": e["op"]}
        if self.use_rank_leak:  # ablation: expose chronological rank + is_min/is_max
            k = len(fracs)
            rank = torch.argsort(torch.argsort(fracs)).float() / max(k - 1, 1)
            is_min = (fracs == lo).float()
            is_max = (fracs == hi).float()
            item["leak"] = torch.stack([rank, is_min, is_max], dim=-1)
        return item


def collate(batch):
    """Pad variable-k candidate sets; return a padding mask (True = pad)."""
    K = max(b["t_rel"].shape[0] for b in batch)
    B = len(batch)
    edim = batch[0]["o_emb"].shape[-1]
    t_rel = torch.zeros(B, K); t_abs = torch.zeros(B, K)
    o_emb = torch.zeros(B, K, edim); r_emb = torch.zeros(B, K, edim)
    valid = torch.zeros(B, K); mask = torch.ones(B, K, dtype=torch.bool)
    q_emb = torch.stack([b["q_emb"] for b in batch])
    anchor = torch.stack([b["anchor"] for b in batch])
    polarity = torch.stack([b["polarity"] for b in batch])
    polarity_iv = torch.stack([b["polarity_iv"] for b in batch])
    has_leak = "leak" in batch[0]
    leak = torch.zeros(B, K, 3) if has_leak else None
    ops = []
    for i, b in enumerate(batch):
        k = b["t_rel"].shape[0]
        t_rel[i, :k] = b["t_rel"]; t_abs[i, :k] = b["t_abs"]
        o_emb[i, :k] = b["o_emb"]; r_emb[i, :k] = b["r_emb"]
        valid[i, :k] = b["valid"]; mask[i, :k] = False
        if has_leak:
            leak[i, :k] = b["leak"]
        ops.append(b["op"])
    out = {"t_rel": t_rel, "t_abs": t_abs, "o_emb": o_emb, "r_emb": r_emb,
           "valid": valid, "mask": mask, "q_emb": q_emb, "anchor": anchor,
           "polarity": polarity, "polarity_iv": polarity_iv, "op": ops}
    if has_leak:
        out["leak"] = leak
    return out


if __name__ == "__main__":  # quick sanity / stats
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    ex = build_examples(args.split)
    ops = Counter(e["op"] for e in ex)
    ksz = Counter(min(len(e["cands"]), 10) for e in ex)
    nvalid = Counter()
    for e in ex:
        nvalid[e["op"]] += sum(c["valid"] for c in e["cands"])
    print(f"{args.split}: {len(ex)} candidate sets  ops={dict(ops)}")
    print(f"  k-dist(capped10)={dict(sorted(ksz.items()))}")
    print(f"  total valids per op={dict(nvalid)}")
