#!/usr/bin/env bash
# Launch one model's pod through models.sh, retrying while no GPU is free (22 Oct load test: 4 of 5 launches found no
# free 80 GB GPU and needed 6-16 min of retries), then start its fetch loop in the background.
# Usage: scripts/pod/launch_retry.sh <model> <out-name> '<run JSON>'      (env: WALL, BUDGET, CONC as for models.sh;
#        TRIES (default 20) x 120 s between attempts; BACKEND=modal launches on Modal instead, see modal_pod.py)
# Never launches twice for one out-name: if runs/<out-name>/pod.json exists, it stops (use pod.py status / fetch).
# Retries ONLY on "no capacity; not launched"; any other failure stops at once (a timeout on POST /pods may already
# have created a pod that pod.json does not know about: check the RunPod console for collusion-<out-name>).
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME=$1; OUTNAME=$2; RUN=$3; TRIES=${TRIES:-20}
RUNS="${HP_MAIN_ROOT:-.}/runs"  # pod.py writes pod.json under HP_MAIN_ROOT/runs when that is set
if [ -e "$RUNS/$OUTNAME/pod.json" ]; then
  echo "$RUNS/$OUTNAME/pod.json exists: a pod was already launched for $OUTNAME; not launching again"; exit 1
fi
if [ "${BACKEND:-pod}" != modal ] && [ -e "$RUNS/$OUTNAME/modal.json" ]; then  # fresh review 24 Oct: never a RunPod pod into a Modal run
  echo "$RUNS/$OUTNAME/modal.json exists: this run belongs to Modal; use modal_pod.py status / fetch"; exit 1
fi
if [ -n "${DRY:-}" ]; then exec bash scripts/pod/models.sh "$NAME" "$OUTNAME" "$RUN"; fi
if [ "${BACKEND:-pod}" = modal ]; then  # Modal (modal_pod.py): one launch (Modal queues for a GPU itself), then its fetch loop
  if [ -e "$RUNS/$OUTNAME/modal.json" ]; then
    echo "$RUNS/$OUTNAME/modal.json exists: already launched on Modal; use modal_pod.py status / fetch"; exit 1
  fi
  bash scripts/pod/models.sh "$NAME" "$OUTNAME" "$RUN" 2>&1 | grep -v -i token
  if [ -e "$RUNS/$OUTNAME/modal.json" ]; then  # also when the App id is not confirmed yet: fetch reconciles it
    PYTHONUNBUFFERED=1 nohup .venv/bin/python scripts/pod/modal_pod.py --out-name "$OUTNAME" fetch --loop 120 \
      > "$RUNS/$OUTNAME.fetch.log" 2>&1 &
    echo "fetch loop: $RUNS/$OUTNAME.fetch.log (keep the laptop awake, e.g. caffeinate)"; exit 0
  fi
  echo "Modal launch for $NAME failed before recording a run: see $RUNS/$OUTNAME/_modal/launch.log"; exit 1
fi
for i in $(seq 1 "$TRIES"); do
  out=$(bash scripts/pod/models.sh "$NAME" "$OUTNAME" "$RUN" 2>&1 | grep -v -i token)
  if [ -e "$RUNS/$OUTNAME/pod.json" ]; then
    echo "$(date +%H:%M:%S) LAUNCHED $NAME as $OUTNAME: $(echo "$out" | grep '^pod ')"
    PYTHONUNBUFFERED=1 nohup .venv/bin/python scripts/pod/pod.py --out-name "$OUTNAME" fetch --loop 60 \
      > "$RUNS/$OUTNAME.fetch.log" 2>&1 &
    echo "fetch loop: $RUNS/$OUTNAME.fetch.log (keep the laptop awake, e.g. caffeinate)"
    exit 0
  fi
  if ! echo "$out" | grep -q "no capacity; not launched"; then
    echo "$out" | tail -20
    echo "$(date +%H:%M:%S) FAILED for $NAME (not a capacity error): not retrying. Check the RunPod console for a pod"
    echo "named collusion-$OUTNAME (a timed-out launch may still have created one; pod.py down cannot see it)."
    exit 1
  fi
  echo "$(date +%H:%M:%S) attempt $i/$TRIES: no capacity for $NAME"
  [ "$i" -lt "$TRIES" ] && sleep 120
done
echo "GAVE UP on $NAME after $TRIES attempts"; exit 1
