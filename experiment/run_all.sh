#!/bin/bash
# Full pipeline; each step logs to experiment/results/logs.
cd "$(dirname "$0")/.." && . .venv/bin/activate && mkdir -p experiment/results/logs
for s in p2_selector p1_diagnostic p3_ablate; do
  echo "== $s $(date)" ; python -u experiment/$s.py > experiment/results/logs/$s.log 2>&1 || { echo "FAILED $s"; exit 1; }
done
echo "ALL DONE $(date)"
