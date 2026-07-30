"""D3.2: ordinal linear probe on timestamp representations.

Does the timestamp representation encode chronological (ordinal) structure? We
train a linear classifier to predict "is date A earlier than date B?" from two
representations of the same trained model:

  (a) special-token embedding  — the single registered `[YYYY-MM-DD]` token row;
  (b) multi-token mean-pool     — mean of the base-vocab subword embeddings of the
      raw "YYYY-MM-DD" string (the pre-registration representation).

If (a) is near chance and (b) is high, the special-token registration *destroyed*
the ordinal structure that raw digit tokens carry (T2, representation-level bias),
which SFT then cannot recover. Directly supports the collapse diagnosis.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TS_FILE = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg", "ts2id.json")


def date_ord(s):
    y, m, d = s.split("-")
    return (int(y) * 13 + int(m)) * 32 + int(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-balanced"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "d3_ordinal_probe.md"))
    args = ap.parse_args()

    timestamps = sorted(json.load(open(TS_FILE)).keys())
    tok = AutoTokenizer.from_pretrained(args.adapter)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
            device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()

    # (a) special-token embedding per timestamp
    spec = np.stack([emb[tok.convert_tokens_to_ids(f"[{ts}]")] for ts in timestamps])
    # (b) multi-token mean-pool of the raw "YYYY-MM-DD"
    multi = []
    for ts in timestamps:
        ids = tok(ts, add_special_tokens=False).input_ids
        multi.append(emb[ids].mean(0))
    multi = np.stack(multi)
    ords = np.array([date_ord(ts) for ts in timestamps])

    rng = np.random.default_rng(0)
    idx = np.arange(len(timestamps))
    ii = rng.choice(idx, args.pairs); jj = rng.choice(idx, args.pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    y = (ords[ii] < ords[jj]).astype(int)

    results = {}
    for name, rep in [("special-token (current)", spec), ("multi-token mean-pool (raw)", multi)]:
        X = np.concatenate([rep[ii] - rep[jj], rep[ii] * rep[jj]], axis=1)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        acc = clf.score(Xte, yte)
        results[name] = acc

    L = ["# D3.2 — Ordinal linear probe on timestamp representations", "",
         f"Model: `{os.path.basename(args.adapter)}`. Task: predict *is A earlier than B?* "
         f"from two representations ({args.pairs} random date pairs, 70/30 split). "
         "Chance = 50%.", "",
         "| representation | earlier-than accuracy |", "|---|--:|"]
    for k, v in results.items():
        L.append(f"| {k} | {100*v:.1f}% |")
    L += ["",
          "## Interpretation",
          "- special-token near chance + multi-token high ⇒ the registered timestamp "
          "tokens carry **no ordinal structure**; raw digit tokens do. The representation, "
          "not the data, blocks `last`/`after` (T2 confirmed).",
          "- If both are high, ordinality is present and the collapse is elsewhere (data/"
          "optimization) — revisit D1/D2."]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
