"""G2: our temporal discriminator as a re-ranker over TimeR4's retrieved facts.

TimeR4 releases, per question, the retrieved "Historical facts" the reader saw
(test_prompt.json). This runner:
  1. parses each verbalized fact back to structured (s,r,o,t) by exact lookup against
     the KG (100% resolvable: TimeR4 verbalizes as "s r o in t" with '_'->' '),
  2. groups facts by (topic-entity, relation) into temporal sibling sets,
  3. scores each candidate timestamp with the trained interval discriminator
     (iv_all.pt), group-locally (the trie-local locus, here the (s,r) group),
  4. either (a) reports a READER-FREE preview (Hits@1 if disc's top pick is the
     answer) entirely on CPU, and/or (b) writes a re-ranked test_prompt.json for the
     GPU reader stage (run TimeR4's predict_answer.py on it), and/or (c) dumps
     per-candidate scores for CPU fusion analysis.

CPU for parsing/scoring/preview (no reader). The reader pass (G1/G2 on 128) consumes
--out_prompts. See results/gpu_experiments_128.md.

Example (CPU preview + write reranked prompts):
  PYTHONPATH=. python discriminator/timer4_rerank.py \
    --prompts timer4/datasets/MultiTQ/prompt/test_prompt.json \
    --kg timer4/datasets/MultiTQ/kg/full.txt \
    --test data/multitq/questions/test.json \
    --disc save_models/discriminator/iv_all.pt \
    --eval_disc --out_prompts timer4/datasets/MultiTQ/prompt/test_prompt_reranked.json \
    --out results/g2_disc_preview.md
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from discriminator.set_encoder import TemporalSetScorer
from discriminator.encode_time import date_to_fracyear
from discriminator.phase2_system_e1 import fine_op, answer_of
from discriminator.phase2_system_hybrid import (
    NEEDS_ANCHOR, mentioned, disc_group_local)
from checker.timepoint import TimePoint
from scripts.build_before_after_signatures import parse_anchor, strict_norm
from scripts.multitq_linker import norm

FACTS_RE = re.compile(r"(Historical facts:\s*)(\[.*?\])(\s*\n)", re.S)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def parse_iso_anchor(q):
    """Parse a numeric ISO date (YYYY-MM-DD / YYYY-MM) from TimeR4's rewritten question.

    parse_anchor() only handles textual months and falls through to a bare-year match,
    so 'Before 2007-10-31' would collapse to 2007.00 (year) and break the interval
    predicate (all candidates fall outside a start-of-year bound). We resolve the full
    date first; parse_anchor() is the fallback for textual forms.
    """
    m = ISO_RE.search(q)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
    if d:
        return TimePoint(y, mo, int(d), "day")
    return TimePoint(y, mo, 1, "month")


def build_kg_vmap(kg_path):
    """verbalized 's r o in t' -> (s,r,o,t) exact map."""
    vmap = {}
    for line in open(kg_path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        s, r, o, t = p[0], p[1], p[2], p[3]
        vmap[f"{s.replace('_',' ')} {r.replace('_',' ')} {o.replace('_',' ')} in {t}"] = (s, r, o, t)
    return vmap


def parse_facts(text, vmap):
    """Return (list of raw fact strings, list of structured (s,r,o,t) or None)."""
    m = FACTS_RE.search(text)
    if not m:
        return [], []
    try:
        raw = ast.literal_eval(m.group(2))
    except Exception:
        return [], []
    struct = [vmap.get(f.rstrip(".").strip()) for f in raw]
    return raw, struct


def build_example(q, struct_facts, rewritten=None):
    """Group TimeR4's retrieved facts by (topic,r,dir) into sibling sets; build cands.

    `rewritten` = TimeR4's rewritten question (from the prompt), which resolves event
    anchors to explicit dates ("After Tony Blair" -> "After 2006-12-07"); we parse the
    anchor and the first/last|before/after surface from it (99% anchor resolution vs 0%
    from the original), falling back to the original question if absent.
    """
    aq = rewritten or q["question"]
    qnorm = norm(q["question"])          # entity mentions: use the original (has names)
    gold = {strict_norm(a) for a in q["answers"]}
    atype, tlevel = q["answer_type"], q["time_level"]
    op = fine_op(q["qtype"], aq)
    facts = [f for f in struct_facts if f is not None]

    # Group key: a sibling set must vary ONLY in the unmentioned slot(s) + time.
    #   both s and o mentioned  -> key (s,r,o): only time varies (answer is a time)
    #   only s mentioned        -> key (s,r): the object varies (answer is the object)
    #   only o mentioned        -> key (o,r): the subject varies (answer is the subject)
    # Grouping by a single entity when BOTH are named pollutes the set with other
    # entities' facts and wrecks first/last (the 2026-07-18 regression).
    def keyof(s, r, o):
        s_in, o_in = mentioned(s, qnorm), mentioned(o, qnorm)
        if s_in and o_in:
            return (s, r, o, "both")
        if s_in:
            return (s, r, "subj")
        if o_in:
            return (o, r, "obj")
        return None

    groups = []
    for (s, r, o, t) in facts:
        k = keyof(s, r, o)
        if k is not None and k not in groups:
            groups.append(k)
    if not groups:  # nothing mentioned -> topic = subject of first fact
        if not facts:
            return None
        s, r, o, t = facts[0]
        groups = [(s, r, "subj")]

    gidx = {g: i for i, g in enumerate(groups)}
    cands = []
    for (s, r, o, t) in facts:
        k = keyof(s, r, o)
        if k is None or k not in gidx:
            continue
        if k[-1] == "both":
            d, ans_o = "both", o          # answer is a time; ans_o unused for time-ans
            fact = (s, r, o, t)
        elif k[-1] == "subj":
            d, ans_o = "subj", o
            fact = (s, r, o, t)
        else:
            d, ans_o = "obj", s
            fact = (s, r, o, t)
        tp = TimePoint.parse(t)
        ans = answer_of(ans_o, t, atype, tlevel)
        cands.append({"o": ans_o, "r": r, "t_str": str(tp), "frac": date_to_fracyear(tp),
                      "valid": int(strict_norm(ans) in gold), "answer": ans,
                      "direction": d, "group": gidx[k], "fact": fact})
    if not cands:
        return None
    topic = groups[0][0]
    anchor_frac = None
    if q["qtype"] in NEEDS_ANCHOR:
        a = parse_iso_anchor(aq) or parse_anchor(aq)   # ISO date first, then textual
        if a is None:
            # fallback: retrieved fact linking topic to the OTHER mentioned entity
            trel = groups[0][1]
            conn = [f for f in facts
                    if mentioned(f[0], qnorm) and mentioned(f[2], qnorm)
                    and norm(f[0]) != norm(f[2])]
            same_rel = [f for f in conn if f[1] == trel]
            pick = (same_rel or conn)
            if pick:
                a = TimePoint.parse(pick[0][3])
        if a is not None:
            anchor_frac = date_to_fracyear(a)
    ex = {"quid": q["quid"], "op": op, "question": q["question"], "s": topic,
          "cands": cands, "anchor_frac": anchor_frac, "qtype": q["qtype"],
          "answer_type": atype, "qlabel": q.get("qlabel", "?"),
          "n_gold_in_set": sum(c["valid"] for c in cands)}
    if q["qtype"] in ("equal", "equal_multi"):
        ex["anchor2_frac"] = anchor_frac
    return ex


def load_disc(path, dev):
    st = torch.load(path, map_location=dev); a = st["args"]
    disc = TemporalSetScorer(h=a.get("h", 256), layers=a.get("layers", 2),
                             time_mode=a.get("time_mode", "bochner"),
                             cond=a.get("cond", "text")).to(dev)
    disc.load_state_dict(st["model"]); disc.eval()
    return disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="timer4/datasets/MultiTQ/prompt/test_prompt.json")
    ap.add_argument("--kg", default="timer4/datasets/MultiTQ/kg/full.txt")
    ap.add_argument("--test", default="data/multitq/questions/test.json")
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/iv_all.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="rerank", choices=["rerank", "filter"],
                    help="rerank: reorder facts by disc score within group; "
                         "filter: keep only disc-top-1 per group (+ ungrouped)")
    ap.add_argument("--eval_disc", action="store_true",
                    help="reader-free preview: Hits@1 taking disc-top candidate as the answer")
    ap.add_argument("--gate", default="",
                    help="comma-sep qtypes to APPLY disc rerank to (others keep retrieval "
                         "order). Empty = apply to all. Recommended: before_after,equal_multi")
    ap.add_argument("--out_prompts", default=None, help="write reranked test_prompt.json")
    ap.add_argument("--dump_scores", default=None)
    ap.add_argument("--out", default=None, help="markdown summary")
    ap.add_argument("-n", type=int, default=0, help="limit #questions (debug)")
    args = ap.parse_args()

    dev = args.device
    print("[load] KG vmap ..."); vmap = build_kg_vmap(args.kg)
    prompts = json.load(open(args.prompts))
    test = json.load(open(args.test))
    assert len(prompts) == len(test), f"{len(prompts)} vs {len(test)}"
    if args.n:
        prompts, test = prompts[:args.n], test[:args.n]
    disc = load_disc(args.disc, dev)

    # build examples (keep index alignment)
    examples, idx_of = [], []
    parsed = []
    for i, (pr, q) in enumerate(zip(prompts, test)):
        raw, struct = parse_facts(pr["text"], vmap)
        parsed.append((raw, struct))
        rw = re.search(r"Question:\s*(.+)$", pr["text"], re.S)
        rq = rw.group(1).strip() if rw else None
        ex = build_example(q, struct, rewritten=rq) if struct else None
        if ex is not None:
            examples.append(ex); idx_of.append(i)
    print(f"[build] {len(examples)}/{len(prompts)} questions have a usable candidate group")

    scores = disc_group_local(disc, examples, dev)   # per-example group-local log-softmax

    # ---- reader-free preview + rerank application ----
    # The honest CPU metric is disc's WITHIN-GROUP temporal selection, not disc-alone
    # over all candidates (that degenerate case throws away retrieval and is known to
    # hurt; the real system fuses disc as an alpha-term). For each question whose gold
    # is in a contested group (>=2 distinct timestamps), we ask: does disc rank a gold
    # candidate top-1 within that group? Baselines: retrieval order, random.
    sel = defaultdict(lambda: [0, 0])          # disc within-group top-1
    sel_first = defaultdict(lambda: [0, 0])    # retrieval-order-in-group top-1
    gate_set = {s.strip() for s in args.gate.split(",") if s.strip()}
    reranked_text = {i: prompts[i]["text"] for i in range(len(prompts))}
    dump = open(args.dump_scores, "w") if args.dump_scores else None

    for ex, sc, i in zip(examples, scores, idx_of):
        q = test[i]
        cds = ex["cands"]
        sc = np.asarray(sc)
        # group membership
        gmembers = defaultdict(list)
        for ci, c in enumerate(cds):
            gmembers[c["group"]].append(ci)
        # gold's contested group(s): >=2 distinct timestamps and contains a valid cand
        for g, idxs in gmembers.items():
            ts = {cds[ci]["t_str"] for ci in idxs}
            has_gold = any(cds[ci]["valid"] for ci in idxs)
            if len(ts) < 2 or not has_gold:
                continue
            disc_top = idxs[int(np.argmax(sc[idxs]))]
            first_top = idxs[0]                       # retrieval order within group
            for k in ("OVERALL", f"q:{q['qtype']}"):
                sel[k][0] += cds[disc_top]["valid"]; sel[k][1] += 1
                sel_first[k][0] += cds[first_top]["valid"]; sel_first[k][1] += 1
        if dump:
            dump.write(json.dumps({"quid": ex["quid"], "qtype": ex["qtype"],
                "answer_type": ex["answer_type"], "qlabel": ex["qlabel"],
                "group": [c["group"] for c in cds],
                "disc": [float(x) for x in sc], "valid": [c["valid"] for c in cds],
                "answer": [c["answer"] for c in cds]}) + "\n")
        order = list(np.argsort(-sc))

        # apply rerank/filter to the fact list -> reranked prompt (gated by qtype)
        if gate_set and q["qtype"] not in gate_set:
            continue                      # leave this prompt at TimeR4's original order
        raw, struct = parsed[i]
        # map each cand to its raw-fact string (by fact tuple)
        fact_str = {}
        for rawf, st in zip(raw, struct):
            if st is not None:
                fact_str[st] = rawf
        cand_raw = []
        for ci in order:
            f = ex["cands"][ci]["fact"]
            if f in fact_str:
                cand_raw.append(fact_str[f])
        if args.mode == "filter":
            # keep disc-top-1 per group + facts not in any candidate group
            keep_top = {}
            for ci in order:
                g = ex["cands"][ci]["group"]
                keep_top.setdefault(g, ex["cands"][ci]["fact"])
            keep_set = set(keep_top.values())
            cand_raw = [fact_str[f] for f in keep_set if f in fact_str]
        used = set(cand_raw)
        rest = [rf for rf in raw if rf not in used]
        new_facts = cand_raw + rest
        new_list = "[" + ", ".join(repr(x) for x in new_facts) + "]"
        reranked_text[i] = FACTS_RE.sub(
            lambda m: m.group(1) + new_list + m.group(3), prompts[i]["text"], count=1)

    if dump:
        dump.close()

    def rate(d, k): return 100 * d[k][0] / max(d[k][1], 1)
    lines = ["# G2 CPU preview: disc within-group temporal selection\n",
             f"questions with candidate group: {len(examples)}/{len(prompts)}\n",
             "Metric: over contested groups (>=2 timestamps, gold present), does the "
             "selector rank a gold candidate top-1 *within the group*? This is disc's "
             "temporal-selection job; the end-to-end answer additionally needs the reader "
             "to pick the group (GPU, --out_prompts).\n",
             "| slice | n | retrieval-order | **disc** | delta |", "|---|---:|---:|---:|---:|"]
    for k in ["OVERALL", "q:after_first", "q:before_last", "q:equal_multi",
              "q:before_after", "q:equal", "q:first_last"]:
        if k in sel:
            n = sel[k][1]; b, d = rate(sel_first, k), rate(sel, k)
            lines.append(f"| {k} | {n} | {b:.1f} | **{d:.1f}** | {d-b:+.1f} |")
    report = "\n".join(lines)
    print("\n" + report)
    print("\nNOTE: within-group selection isolates the temporal repair from retrieval. "
          "End-to-end Hits@1 comes from the reader on --out_prompts (GPU, server 128).")

    if args.out_prompts:
        out = [{"text": reranked_text[i], "answers": prompts[i]["answers"],
                "question": prompts[i]["question"]} for i in range(len(prompts))]
        json.dump(out, open(args.out_prompts, "w"))
        print(f"[write] reranked prompts -> {args.out_prompts}")
    if args.out:
        open(args.out, "w").write(report + "\n")
        print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
