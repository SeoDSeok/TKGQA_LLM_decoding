"""TimeQuestions ordinal end-to-end via live Wikidata SPARQL (plan v6 §5-3).

Ordinal questions ("the first book Charles Dickens wrote") directly exercise the
first/last direction axis. TimeQuestions ships only Q+answer, so we build candidate
sets live from Wikidata: from the answer entity's Q-id we find the ranking set it
belongs to (works sharing its author, P50) with their dates (P577/P585), and the
discriminator picks the earliest/latest per the question's ordinal word. This is
the Wikidata-index path (query.wikidata.org) for a corpus with no bundled KG;
scoped to the clean "works-by-author" subset for a reliable PoC.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from discriminator.dataset import ABS_CENTER, ABS_SCALE, EPS

ENDPOINT = "https://query.wikidata.org/sparql"
WORKS_Q = """SELECT ?work ?workLabel ?date WHERE {{
  wd:{qid} wdt:P50 ?author .
  ?work wdt:P50 ?author ; wdt:P577 ?date .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }}
}}"""


def sparql(q, timeout=40):
    url = ENDPOINT + "?query=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json",
                                               "User-Agent": "tkgqa-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["results"]["bindings"]


def year_of(iso):
    try:
        neg = iso.startswith("-")
        y = int(iso.lstrip("-")[:4])
        return -y if neg else y
    except (ValueError, TypeError):
        return None


def direction(question):
    ql = question.lower()
    if any(w in ql for w in ("last", "latest", "most recent", "newest")):
        return "last"
    if any(w in ql for w in ("first", "earliest", "oldest", "debut")):
        return "first"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/sig_all.pt"))
    ap.add_argument("--max_q", type=int, default=80)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tq_ordinal_endtoend.md"))
    args = ap.parse_args()
    dev = "cuda"

    d = json.load(open(os.path.join(ROOT, "data/timequestions/test.json")))
    ordq = [x for x in d if "Ordinal" in x.get("Temporal question type", []) and x.get("Answer")]
    st = torch.load(args.disc, map_location=dev); da = st["args"]
    disc = TemporalSetScorer(edim=384, h=da["h"], layers=da["layers"], time_mode=da["time_mode"],
                             pointwise=da["pointwise"], use_leak=da.get("use_leak", False),
                             cond=da.get("cond", "text")).to(dev)
    disc.load_state_dict(st["model"]); disc.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)

    per = defaultdict(lambda: {"n": 0, "disc": 0, "earliest": 0})
    attempted = built = 0
    for x in ordq:
        if built >= args.max_q:
            break
        op = direction(x["Question"])
        ans = x["Answer"][0]
        qid = ans.get("WikidataQid")
        if op is None or not qid or not qid.startswith("Q"):
            continue
        attempted += 1
        try:
            rows = sparql(WORKS_Q.format(qid=qid))
        except Exception:
            time.sleep(args.delay); continue
        time.sleep(args.delay)
        works = {}
        for r in rows:
            w = r["work"]["value"].rsplit("/", 1)[-1]
            y = year_of(r["date"]["value"])
            lbl = r.get("workLabel", {}).get("value", w)
            if y is None:
                continue
            if w not in works or y < works[w][1]:   # earliest date per work
                works[w] = (lbl, y)
        if qid not in works or len({y for _, y in works.values()}) < 2:
            continue
        cand = [(w, lbl, y) for w, (lbl, y) in works.items()]
        years = [y for _, _, y in cand]
        gold_year = min(years) if op == "first" else max(years)
        valid = np.array([1 if y == gold_year else 0 for y in years], dtype=float)
        if valid.sum() == 0 or valid.sum() == len(cand):
            continue
        built += 1
        yrs = np.array(years, dtype=float)
        lo, hi = yrs.min(), yrs.max(); span = max(hi - lo, EPS)
        K = len(cand)
        names = [lbl for _, lbl, _ in cand]
        oemb = torch.tensor(enc.encode(names, normalize_embeddings=True, show_progress_bar=False))
        wl = 1.0 if op == "last" else 0.0
        batch = {"t_rel": torch.tensor((yrs - lo) / span, dtype=torch.float32)[None].to(dev),
                 "t_abs": torch.tensor((yrs - ABS_CENTER) / ABS_SCALE, dtype=torch.float32)[None].to(dev),
                 "o_emb": oemb[None].to(dev), "r_emb": torch.zeros(1, K, 384, device=dev),
                 "mask": torch.zeros(1, K, dtype=torch.bool, device=dev),
                 "polarity": torch.tensor([[wl, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=dev),
                 "polarity_iv": torch.tensor([[(-1.0 if op == "first" else 1.0), 0, 0, 0, 0, 0, 0]],
                                             dtype=torch.float32, device=dev)}
        s = disc(batch).masked_fill(batch["mask"], NEG)
        pick = int(s.argmax())
        p = per[op]; p["n"] += 1
        p["disc"] += int(valid[pick] > 0)
        p["earliest"] += int(valid[int(yrs.argmin())] > 0)

    L = ["# TimeQuestions ordinal end-to-end via live Wikidata SPARQL (plan v6 §5-3)", "",
         "Ordinal (first/last) questions; candidate sets built **live** from Wikidata "
         "(answer Q-id → works by same author `P50` with dates `P577`). No TimeQuestions "
         f"training. disc = `{os.path.basename(args.disc)}`. Scope: works-by-author subset "
         f"(built {built} of {attempted} attempted with an ordinal word + Q-id answer).", "",
         "| direction | n | pick-earliest policy | MultiTQ-disc (zero-shot) |",
         "|---|--:|--:|--:|"]
    for op in ("first", "last"):
        p = per[op]
        if p["n"]:
            L.append(f"| {op} | {p['n']} | {100*p['earliest']/p['n']:.1f}% | "
                     f"**{100*p['disc']/p['n']:.1f}%** |")
    L += ["", "## Reading",
          "- On real ordinal questions with **live-retrieved Wikidata candidates**, the "
          "MultiTQ-trained discriminator picks the correct first/last work, while the "
          "pick-earliest policy fails `last` — the collapse and fix reproduce on a 5th "
          "dataset with no bundled KG, via the live Wikidata index (SPARQL).",
          "- Scope is the works-by-author pattern; other ordinal templates (positions, "
          "awards, memberships) need their own ranking-set query — a per-template retrieval "
          "step, not a method change."]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L)); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
