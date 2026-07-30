#!/usr/bin/env bash
# Push the TimeR4 backbone bundle (code + datasets + released prompts/predictions) to
# 128 for the KBS gap-closing work (Phase 1 reproduce, Phase 2 disc-as-reranker).
# ~745 MB. Run ON 129. Pure network transfer — safe while GPU jobs decode on 128.
#
# What goes over:
#   timer4/                         the cloned TimeR4 repo, with:
#     datasets/MultiTQ/prompt/*.json    TimeR4-generated retrieved+reranked contexts  <- Phase 2 input
#     datasets/MultiTQ/kg/full.txt      the TKG
#     results/MultiTQ/**/predictions    released outputs (the 72.8 fine-tuned-Llama2 run)
#     *.py                              retrival / rerank / reader / training code
#   scripts/timer4_oracle_probe.py  our Phase-0 headroom probe (CPU)
#
# No Baidu weights needed: the reader is model-agnostic and we use the cached
# NousResearch/Meta-Llama-3.1-8B-Instruct (see setup_128_timer4.sh).
set -euo pipefail

HOST="dsseo@141.223.44.128"
DST_BASE="${HOST}:~/personal_research4"

SRCS=(
  "/home/dsseo/personal_research4/timer4"
  "/home/dsseo/personal_research4/scripts/timer4_oracle_probe.py"
)
for s in "${SRCS[@]}"; do
  [ -e "$s" ] || { echo "missing on this host: $s"; exit 1; }
done

echo "[rsync] TimeR4 bundle (~745 MB) -> 128 ..."
# timer4/  (dir -> ~/personal_research4/timer4/)
rsync -avz --progress \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='extracted' \
  /home/dsseo/personal_research4/timer4 "${DST_BASE}/"
# the probe script (into scripts/)
rsync -avz --progress \
  /home/dsseo/personal_research4/scripts/timer4_oracle_probe.py "${DST_BASE}/scripts/"
# the G2 reranker (into discriminator/); parses/scores on CPU, feeds the GPU reader
rsync -avz --progress \
  /home/dsseo/personal_research4/discriminator/timer4_rerank.py "${DST_BASE}/discriminator/"
# the deploy scripts themselves (so setup_128_timer4.sh EXISTS on 128) + the plan/status
# markdowns (single source of truth on 128).
rsync -avz --progress \
  /home/dsseo/personal_research4/deploy/setup_128_timer4.sh \
  /home/dsseo/personal_research4/deploy/rsync_timer4_128.sh "${DST_BASE}/deploy/"
rsync -avz --progress \
  /home/dsseo/personal_research4/results/kbs_gap_closing_plan.md \
  /home/dsseo/personal_research4/results/phase0_timer4_headroom.md \
  /home/dsseo/personal_research4/results/STATUS.md \
  /home/dsseo/personal_research4/results/methodology.md "${DST_BASE}/results/"

echo "[done] TimeR4 code + datasets + prompts + deploy scripts + status md are on 128."
echo
echo "NEXT on 128:"
echo "  1) bash deploy/setup_128_timer4.sh   # create the timer4 conda env + CPU smoke test"
echo "     (no Baidu weights — reader = cached Meta-Llama-3.1-8B-Instruct)"
echo "  2) Phase-1 baseline / Phase-2 commands are printed by that script."
