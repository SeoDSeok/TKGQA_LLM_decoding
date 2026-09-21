# Structurally faithful, temporally blind

Code for **"Structurally faithful, temporally blind: diagnosing temporal collapse
in knowledge-graph-constrained LLM decoding."**

Graph-constrained decoding restricts an LLM to valid KG paths, guaranteeing
**structural** validity — but not **temporal** validity. We diagnose a systematic
*earliest-timestamp bias* in a GCR-style decoder (near-100% structural faithfulness
while temporal validity on later-directed operators collapses to ~2%), show it is
**strongly associated with chronological special-token registration in our
implementation**, and repair it with a lightweight **value-space temporal
discriminator** fused *trie-locally* at the timestamp branch. We then delimit when
the repair helps: across the settings we measured, its benefit grows with the
backbone's temporal blindness.

## Results at a glance

| | |
|---|---|
| Collapse | `first`/`before` ~98% TVR, `last`/`after` ~2%, SFR ~100% throughout |
| Repair (k-hop, controlled) | 44.7% → **99.3%** answer accuracy |
| Global mixed-set control | 14.7%, with (relation, object) branch fidelity 23.3% |
| Full-test MultiTQ | Hits@1 30.1 → **39.2** (paired, identical front-end/backbone) |
| Discriminator | **2.08M trainable** params over a frozen MiniLM encoder |
| Added cost | **no LLM forward pass, no external API call**; one 2.08M-param forward per contested branch (2.7–6.3 ms, CPU) |

### Which factor makes trie-local integration work

The global control changes two things at once, so we ran a factorial control
(one run, three arms, identical LLM scores / candidate set / discriminator params —
`discriminator/locus_factorial.py`, `results/locus_factorial.json`):

| arm | scoring context | normalization | Hits@1 @ α=0.2 |
|---|---|---|--:|
| **A** | per-group | per-group | **99.3** |
| B | per-group | global | 98.7 |
| C | mixed-set | global | 14.7 |

A vs B isolates normalization scope (−0.6); B vs C isolates scoring context (−84.0).
So **branch-local candidate conditioning is what makes the method work**, while
per-group normalization mainly buys robustness to the fusion weight (the A–B gap
grows to 41.7 points at α=5). We do *not* claim the arms differ only in where an
otherwise identical score is injected.

## Honest scope

- **The full-test MultiTQ result is not a raw-question end-to-end system.** The
  coarse predicate family is read from the benchmark's question-type field and the
  direction from a deterministic keyword rule; the topic entity, relation, anchor
  and candidate set are reconstructed from the question. A corpus without such a
  field would need a predicate classifier, whose errors would propagate.
- **No SOTA claim.** The number to weigh is the paired 30.1 → 39.2 on an identical
  front-end, candidate set and backbone — not a position in a leaderboard. Systems
  we compare against differ in supervision, front-end assumptions and cost class.
- **The registration finding is a correlate, not an isolated cause.** The
  shuffled-registration run uses a single permutation and a single training seed,
  and permuting registration order also permutes embedding-row assignment,
  initialization neighbourhood and update order. Nothing in the method depends on
  the attribution: all three timestamp representations tested leave temporal
  validity at or below 46.2%.
- **Two experimental forms.** *Token-level* in-decoder fusion (CronQuestions,
  TimelineKGQA) applies the boost at the timestamp branch through a logits
  processor; *candidate-level* selection (full MultiTQ, injection-locus analysis)
  enumerates the trie-admissible facts and ranks them. Both are restricted by the
  same KG-Trie.
- **Shared-timestamp pooling.** The pooling equation in the paper binds only in the
  token-level form; the released token-level runs predate it and write a single tied
  candidate's score. Bounded there (the answer is the timestamp itself); no MultiTQ
  number is affected, since candidate-level selection has no shared scoring slot.
- **Held-out operators** are reused compositionally *under a supplied interval
  specification* — not zero-shot induction of operator semantics from text.

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
                 #   locus_factorial.py = factorial injection-locus control (A/B/C)
decoding/        # trie-local logits-fusion LogitsProcessor
eval/            # TVR/CVR/SFR/Hits@1 aggregation + baseline runners
configs/         # operator map, etc.
scripts/         # operator map / KG index builders, tokenizer registration + audits
tests/           # unit tests
timer4/          # TimeR4 plug-in experiment (code only; datasets/results excluded)
patches/         # local patch(es) to the external GCR baseline (see below)
results/         # metrics (JSON), figures, and figure-generation scripts
                 #   locus_factorial.json + make_locus_figure.py = the A/B/C control
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
- `save_models/` — LoRA adapters / discriminator checkpoints (tens of GB).
  Available from the corresponding author on request; the training scripts and
  configurations needed to regenerate them from the public benchmarks are included.
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

Reproducing the injection-locus control (needs the GCR adapter and a trained
discriminator, see *Not included* below):
```bash
PYTHONPATH=. python discriminator/locus_factorial.py --per_op 150
python results/make_locus_figure.py
```
Run everything from the repo root with `PYTHONPATH=.`.

## Citation
If you use this code, please cite the accompanying paper.

## License
Released under the [MIT License](LICENSE). The external GCR baseline is not
included and remains under its own license.
