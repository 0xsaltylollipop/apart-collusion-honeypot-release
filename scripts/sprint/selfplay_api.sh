#!/usr/bin/env bash
# Sprint self-play answer key, API models (docs/RUNBOOK.md section 2): 3 firms, chat, 300 rounds, 6 games, seeds 1001-1006.
# Runs in the background; rerunning resumes (finished rounds are skipped). Logs: runs/sprint-sp-<model>.log
set -euo pipefail
cd "$(dirname "$0")/../.."
SAMP='{"reasoning_effort": "medium", "reasoning_summary": "auto", "max_tokens": 16384, "temperature": 1.0}'
for m in gpt-6-luna claude-haiku-5-5 claude-sonnet-5-5 gpt-6.1-sol; do
  nohup caffeinate -i bash -c "for try in 1 2 3 4 5; do
    .venv/bin/python -m collusion selfplay --model $m --out runs/sprint-sp-$m/selfplay/f3-chat \
      --games 6 --periods 300 --seed0 1001 --firms 3 --chat --concurrency 6 --sampling '$SAMP' \
      --tag sprint && break; sleep 60; done" >> runs/sprint-sp-$m.log 2>&1 &
  echo "started $m -> runs/sprint-sp-$m.log"
done
