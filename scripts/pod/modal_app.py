"""Modal as the fallback GPU source for the open-model honeypot runs (24 Oct: RunPod had no CUDA-13 hosts for hours).

The same vLLM image as the RunPod pods (pod.IMAGE), the same collusion package and the same runner (runner.py, with
platform "modal": rate from the config, no laptop ack, no self-termination), so the subjects are served and sampled
exactly as on a pod. Results go to the Volume "collusion-runs" under <out-name>/runs/, in the layout the pod serves as
results.tgz; model weights are cached in the Volume "collusion-hf-cache".

Launch it only through modal_pod.py (it writes the run config, picks the GPU and passes the key names):

    python scripts/pod/modal_pod.py --out-name <name> launch <the pod.py launch flags> [--gpu H100]
"""

import json
import os
import secrets
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] if modal.is_local() else Path("/")  # in the container this file sits at /root/, no checkout
IMAGE = "vllm/vllm-openai:v0.31.0"  # = pod.IMAGE (tests/test_modal_fallback.py checks they agree)
RUNS_VOLUME = "collusion-runs"
HF_VOLUME = "collusion-hf-cache"
WORK = "/vol"  # the runs Volume's mount point; each run works in /vol/<out-name>
BOOT = "/root/boot"  # runner.py and the collusion package (HARNESS_DIR = /root/boot/code)
MAX_TIMEOUT_S = 24 * 3600  # Modal's maximum function timeout (docs: guide/timeouts)

app = modal.App(os.environ.get("HP_MODAL_APP", "collusion-honeypot"))
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)

# The vLLM image's own entrypoint (`vllm serve`) is cleared, as in Modal's vLLM examples; the runner starts vLLM itself.
# HP_MODAL_ADD_PYTHON=3.12 adds Modal's standalone Python if the image's python3 does not suit Modal's runtime (the
# runner then calls the image's `vllm` CLI from PATH and pip-installs the harness's few dependencies, as on a pod).
_add_python = os.environ.get("HP_MODAL_ADD_PYTHON") or None
_skip = ["__pycache__", "**/__pycache__/**", "*.pyc"]
image = (modal.Image.from_registry(IMAGE, add_python=_add_python)
         .entrypoint([])
         .env({"HF_HOME": "/root/.cache/huggingface"})  # the weights cache Volume (mounted there below)
         .add_local_dir(HERE, BOOT, ignore=_skip)
         .add_local_dir(CODE / "collusion", f"{BOOT}/code/collusion", ignore=_skip)
         .add_local_file(CODE / "pyproject.toml", f"{BOOT}/code/pyproject.toml"))


@app.function(image=image, gpu="L40S", timeout=MAX_TIMEOUT_S, cpu=8.0, memory=32768,
              volumes={WORK: runs_vol, "/root/.cache/huggingface": hf_vol})
def run_pod(cfg_json: str, out_name: str) -> str:
    """One model's run, as runner.main() does it on a pod. Returns the final phase (done | timeout | error)."""
    os.environ.update({"POD_WORKSPACE": f"{WORK}/{out_name}", "HP_CFG": cfg_json,
                       "RESULTS_TOKEN": secrets.token_urlsafe(24), "HARNESS_DIR": f"{BOOT}/code"})
    sys.path.insert(0, BOOT)
    import runner  # noqa: PLC0415  (reads its env at import)
    try:
        return runner.main()
    finally:
        runs_vol.commit()


def _keys(names: list[str]) -> dict[str, str]:
    """The named keys from the main checkout's .env (the OpenAI keys for firm C and its checker), never printed."""
    path = Path(os.environ.get("HP_MAIN_ROOT", str(CODE))) / ".env"
    vals = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    return {k: vals[k] for k in names if vals.get(k)}


@app.local_entrypoint()
def main(out_name: str, cfg_file: str, gpu: str = "L40S", timeout_h: float = 23.5, keys: str = ""):
    """Runs on the laptop under `modal run --detach` (started by modal_pod.py): records the App id next to the run
    config (modal_pod.py down needs it), then runs the function with this launch's GPU, timeout and keys."""
    cfg_path = Path(cfg_file)
    (cfg_path.parent / "app.json").write_text(json.dumps({"app_id": app.app_id, "gpu": gpu}))
    fn = run_pod.with_options(gpu=gpu, timeout=min(MAX_TIMEOUT_S, int(timeout_h * 3600)),
                              secrets=[modal.Secret.from_dict(_keys([k for k in keys.split(",") if k]))])
    phase = fn.remote(cfg_path.read_text(), out_name)
    print(f"modal run finished: phase={phase}", flush=True)
