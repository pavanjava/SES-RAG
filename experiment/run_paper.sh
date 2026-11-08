#!/bin/bash
cd "$(dirname "$0")/.." && . .venv/bin/activate
for job in "hotpot" "hotpot nomention" "musique" "2wiki"; do
  tag=${job// /_}; echo "== $job $(date)"
  python -u experiment/paper_run.py $job > experiment/results/logs/paper_$tag.log 2>&1 || { echo "FAILED $job"; exit 1; }
done
echo "ALL DONE $(date)"
