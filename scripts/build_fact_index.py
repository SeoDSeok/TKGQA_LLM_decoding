"""Dense fact index for the TimeR4-style retrieval front-end (no entity linking).

Verbalizes every MultiTQ quadruple ("s r o on t", underscores -> spaces) and embeds
it with frozen all-MiniLM-L6-v2 (the same encoder the discriminator uses), saving a
normalized fp16 matrix + the fact list. Retrieval is then cosine top-N by matmul.
ICEWS has no entity-linking tool (TimeR4 §4.2), so dense retrieval supplies
candidates for EVERY question — this replaces the string-match linker front-end.

GPU: ~5 min for 461k facts. CPU: ~30-60 min (use --limit for dry-runs).
"""
import argparse
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KG = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg")
OUT = os.path.join(ROOT, "data", "multitq", "index", "fact_dense.pkl")


def verbalize(s, r, o, t):
    return f"{s.replace('_', ' ')} {r.replace('_', ' ').lower()} {o.replace('_', ' ')} on {t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="cap #facts (CPU dry-run)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    facts = []
    with open(f"{KG}/full.txt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                facts.append(tuple(p))   # (s, r, o, t)
            if args.limit and len(facts) >= args.limit:
                break
    print(f"facts: {len(facts)}")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    texts = [verbalize(*f) for f in facts]
    emb = enc.encode(texts, batch_size=args.batch, convert_to_numpy=False,
                     convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=True)
    emb = emb.cpu().to(torch.float16)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"facts": facts, "emb": emb}, f)
    print(f"wrote {args.out}  ({emb.shape[0]}x{emb.shape[1]} fp16, "
          f"{emb.numel()*2/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
