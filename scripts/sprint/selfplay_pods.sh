#!/usr/bin/env bash
# Sprint self-play answer key, open models on RunPod (docs/RUNBOOK.md section 3). Budgets: The author, 23 Oct.
# Each launch retries while no GPU is free and then starts its fetch loop. Logs: runs/sprint-sp-<name>.launch.log
set -euo pipefail
cd "$(dirname "$0")/../.."
nohup caffeinate -i -t 100000 >/dev/null 2>&1 &   # keep the laptop awake for the fetch loops (~28 h)
RUN='{"games":0,"selfplay":[{"name":"f3-chat","firms":3,"chat":true,"periods":300,"games":6,"seed0":1001,"tag":"sprint"}]}'
launch() { WALL=$2 BUDGET=$3 nohup scripts/pod/launch_retry.sh "$1" "$4" "$5" > "runs/$4.launch.log" 2>&1 & echo "launching $4"; }
launch nemo 24 65 sprint-sp-nemo "$RUN"; sleep 20     # slowest first
launch oss   4 11 sprint-sp-oss  "$RUN"
launch q38  12 33 sprint-sp-q38  "$RUN"
launch muse 10 27 sprint-sp-muse "$RUN"
for s in 1001 1003 1005; do   # Gemma: 3 pods x 2 games (KV cache)
  launch gemma 10 27 "sprint-sp-gemma-$s" \
    '{"games":0,"selfplay":[{"name":"f3-chat","firms":3,"chat":true,"periods":300,"games":2,"seed0":'$s',"tag":"sprint"}]}'
done
