#!/usr/bin/env bash
# Pod bootstrap. Usage: WANT_TEACHER=1 bash setup.sh   (teacher 14B only on the W pod, or for judge fallback)
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p /workspace/models /workspace/data /workspace/out
cd /workspace/persona-rl
pip install -q "vllm==0.31.0" 2>&1 | tail -3
pip install -q "trl==1.16.0" "peft==0.21.2" "transformers==5.17.0" "datasets>=4.7.0" "accelerate>=1.11.0" "openai>=1.50" numpy pytest 2>&1 | tail -3
pip check || echo "WARN: pip check reported conflicts (see above)"
python - <<'PY'
import torch, vllm, trl, peft, transformers, datasets
print("torch", torch.__version__, "| vllm", vllm.__version__, "| trl", trl.__version__, "| peft", peft.__version__, "| transformers", transformers.__version__)
assert torch.__version__.startswith("2.13"), "torch pin broken: " + torch.__version__
PY
python - <<'PY'
import os
from huggingface_hub import snapshot_download
ids = {"qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen1p5b": "Qwen/Qwen2.5-1.5B-Instruct"}
if os.environ.get("WANT_TEACHER") == "1":
    ids["qwen14b"] = "Qwen/Qwen2.5-14B-Instruct"
for name, rid in ids.items():
    p = snapshot_download(rid, local_dir=f"/workspace/models/{name}", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py"])
    print("downloaded", rid, "->", p)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv
pip freeze > /workspace/out/pip_freeze.txt
echo SETUP_OK
