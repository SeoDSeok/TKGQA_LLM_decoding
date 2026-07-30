#!/usr/bin/env bash
# Push the transfer-KG datasets (excluded from the E1 minimal bundle) to 128, so the
# overnight multi-dataset scaling (cronq / tlkgqa / tiq) can run. ~350 MB total.
# Run ON 129. Prompts for the 128 password once (ssh-copy-id to skip). Pure network
# transfer — safe to run while GPU jobs decode on 128.
set -euo pipefail

DST="dsseo@141.223.44.128:~/personal_research4/data/"
SRCS=(
  "/home/dsseo/personal_research4/data/cronquestions"    # cronq_transfer.py   (247 MB)
  "/home/dsseo/personal_research4/data/timelinekgqa"     # tlkgqa_transfer.py  (87 MB)
  "/home/dsseo/personal_research4/data/wikidata_index"   # tiq_endtoend.py: KG index (16 MB)
  "/home/dsseo/personal_research4/data/tiq"              # tiq_endtoend.py: questions via load_tiq (21 MB)
)
for s in "${SRCS[@]}"; do
  [ -e "$s" ] || { echo "missing on this host: $s"; exit 1; }
done
echo "[rsync] transfer-KG datasets (~350 MB) -> 128 ..."
rsync -avz --progress "${SRCS[@]}" "$DST"
echo "[done] cronq / tlkgqa / tiq inputs are on 128. Now runnable:"
echo "  python discriminator/cronq_transfer.py  --per 5000 --out results/cronq_transfer_full.md"
echo "  python discriminator/tlkgqa_transfer.py --per 5000 --out results/tlkgqa_transfer_full.md"
echo "  python discriminator/tiq_endtoend.py    --per 2000 --out results/tiq_endtoend_full.md"
