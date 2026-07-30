# Structurally faithful, temporally blind

Code for **"Structurally faithful, temporally blind: diagnosing temporal collapse
in knowledge-graph-constrained LLM decoding and delimiting its repair across
backbones."**

Graph-constrained decoding restricts an LLM to valid KG paths, guaranteeing
**structural** validity — but not **temporal** validity. We diagnose a systematic
*earliest-timestamp bias* in a GCR-style decoder (near-100% structural faithfulness
while temporal validity on later-directed operators collapses to ~2%), trace it to
chronological special-token registration, and repair it with a lightweight
**value-space temporal discriminator** fused *trie-locally* at the timestamp branch.
We then delimit when the repair helps: its benefit scales monotonically with the
backbone's temporal blindness.

## Metrics (Section 3)
- **TVR** — Temporal Validity Rate: the selected timestamp satisfies the question's
  temporal operator (`first/last/before/after/equal`), reported per operator.
- **SFR** — Structural Faithfulness Rate: the decoded path is a real KG quadruple.
- **CVR** — Conjunctive Validity Rate: TVR ∧ SFR (structurally *and* temporally correct).

## Layout
```
checker/         # evaluation-only TVR / SFR / CVR checker (rule-based, unit-tested)
discriminator/   # value-space temporal discriminator: Bochner time encoding,
                 #   interval predicate, question-conditioned set encoder, training/eval
decoding/        # trie-local logits-fusion LogitsProcessor
eval/            # TVR/CVR/SFR/Hits@1 aggregation + baseline runners
configs/         # operator map, etc.
scripts/         # operator map / KG index builders, tokenizer registration + audits
tests/           # unit tests
timer4/          # TimeR4 plug-in experiment (code only; datasets/results excluded)
patches/         # local patch(es) to the external GCR baseline (see below)
results/         # metrics (JSON), figures, and figure-generation scripts
```

## External baseline: GCR
The graph-constrained decoder is the **GCR** baseline
([RManLuo/graph-constrained-reasoning](https://github.com/RManLuo/graph-constrained-reasoning)),
which is **not re-hosted here**. To reproduce:

```bash
git clone https://github.com/RManLuo/graph-constrained-reasoning.git gcr_base
cd gcr_base && git checkout 9518e8e      # commit this work was built on
# apply our single compatibility patch (transformers>=5 quantization kwargs):
git apply ../patches/gcr_base__base_hf_causal_model.patch
```

## Not included in this repository
To keep the repo code-only, the following are **git-ignored** and must be obtained
or regenerated separately:
- `save_models/` — LoRA adapters / checkpoints (tens of GB)
- `data/` — MultiTQ, CronQuestions, TimelineKGQA, TIQ indices, cached text embeddings
- `timer4/datasets/`, `timer4/results/` — TimeR4 prompts and prediction dumps
- model/data binaries (`*.pt`, `*.pkl`, `*.pickle`, `*.safetensors`, …) and large archives

MultiTQ and CronQuestions are public; download them from their original releases and
place them under `data/`.

## Quick start
```bash
PYTHONPATH=. python -m pytest -q                 # run the checker unit tests
PYTHONPATH=. python scripts/build_operator_map.py
PYTHONPATH=. python discriminator/rule_baseline.py   # parameter-free comparator baseline
```
Run everything from the repo root with `PYTHONPATH=.`.

## Citation
If you use this code, please cite the accompanying paper.

## License
Released under the [MIT License](LICENSE). The external GCR baseline is not
included and remains under its own license.
