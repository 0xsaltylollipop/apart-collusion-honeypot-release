#!/usr/bin/env bash
# Launch one self-terminating pod for an open-weights model with its exact serving settings (pinned revisions).
# Usage: scripts/pod/models.sh <model> <out-name> '<run JSON for --trap>'     (DRY=1 for a dry run; CONC=5 caps games;
#        BACKEND=modal launches on Modal instead of RunPod, MODAL_GPU overrides its GPU)
#   e.g. scripts/pod/models.sh oss f3chat-oss '{"games":0,"selfplay":[{"name":"f3-chat","firms":3,"chat":true,"periods":300,"games":6,"seed0":101}]}'
# The settings are the ones of the 21 Oct self-play grid (docs/reviews/selfplay-grid-2026-10-21.md). R1-Distill-Qwen-32B
# was dropped on 21 Oct (docs/DECISIONS.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
NAME=$1; OUTNAME=$2; RUN=$3
GPU80='["NVIDIA H100 80GB HBM3","NVIDIA H100 PCIe","NVIDIA H100 NVL","NVIDIA RTX PRO 6000 Blackwell Server Edition","NVIDIA RTX PRO 6000 Blackwell Workstation Edition","NVIDIA A100 80GB PCIe","NVIDIA A100-SXM4-80GB"]'
BUDGET=${BUDGET:-30}; WALL=${WALL:-16}; CONC=${CONC:-}  # CONC: games in flight (streams = CONC x 3 self-play, x 2 honeypot)
common=(--max-tokens 32768 --max-model-len 65536 --budget-usd "$BUDGET" --wall-hours "$WALL" --trap "$RUN"
        ${CONC:+--concurrency "$CONC"})
case $NAME in
  muse) M=(--hf-model RedHatAI/Muse-Glimmer-30B-FP8-block --served muse-glimmer-30b-fp8
           --revision 1deb4641ff84f9a728dd11b27cac1f6a02a9ed14 --parser muse_glimmer
           --chat-template-kwargs '{"reasoning_strength": "medium"}' --gpus "$GPU80") ;;
  q38) M=(--hf-model Qwen/Qwen3.8-27B-FP8 --served qwen3.8-27b-fp8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
          --parser qwen3 --extra-vllm '["--language-model-only"]' --chat-template-kwargs '{"reasoning_effort": "medium"}'
          --gpus "$GPU80") ;;
  gemma) M=(--hf-model RedHatAI/gemma-4-31B-it-FP8-block --served gemma-4-31b-it-fp8-block
            --revision d7242548c457ab4b45bd3adb8937f2659af4739d --parser gemma4
            --chat-template-kwargs '{"enable_thinking": true}' --gpus "$GPU80") ;;
  oss) M=(--hf-model openai/gpt-oss-20b --served gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee
          --parser openai_gptoss --gpus "$GPU80") ;;
  nemo) M=(--hf-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --served nemotron-3.5-lightning-30b-a3b
           --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 --parser nemotron_v3
           --extra-vllm '["--mamba-ssm-cache-dtype", "float16"]' --chat-template-kwargs '{"enable_thinking": true}'
           --gpus "$GPU80" --disk-gb 150) ;;
  *) echo "unknown model $NAME (muse, q38, gemma, oss, nemo)"; exit 2 ;;
esac
if [ "${BACKEND:-pod}" = modal ]; then  # 24 Oct: Modal as the fallback GPU source (scripts/pod/modal_pod.py)
  # H100 by default: the same 80 GB class as the self-play pods (GPU80); MODAL_GPU=L40S is cheaper for gpt-oss-20b
  $PY scripts/pod/modal_pod.py --out-name "$OUTNAME" launch "${M[@]}" "${common[@]}" --gpu "${MODAL_GPU:-H100}" \
    ${DRY:+--dry-run}
else
  $PY scripts/pod/pod.py --out-name "$OUTNAME" launch "${M[@]}" "${common[@]}" ${DRY:+--dry-run}
fi
