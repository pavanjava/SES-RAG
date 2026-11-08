#!/bin/bash
cd "$(dirname "$0")/.." && . .venv/bin/activate && mkdir -p experiment/results/logs
for job in "confirm.py:hotpot" "confirm.py nomention:hotpot_nomention" "confirm.py musique:musique"; do
  cmd=${job%%:*}; tag=${job##*:}
  echo "== $cmd $(date)"
  python -u experiment/$cmd > experiment/results/logs/confirm_$tag.log 2>&1 || { echo "FAILED $cmd"; exit 1; }
  python -u experiment/analyze.py $tag > experiment/results/logs/analyze_$tag.log 2>&1 || { echo "FAILED analyze $tag"; exit 1; }
done
echo "ALL DONE $(date)"
