#!/usr/bin/env bash
# RunPod Serverless：准备 Volume checkpoints → 预热模型 → handler
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
if [[ "${VOLUME_ROOT}" == "/models" && -d /runpod-volume ]]; then
  VOLUME_ROOT=/runpod-volume
fi
export MODEL_VERSION="${MODEL_VERSION:-2.5}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${VOLUME_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${VOLUME_ROOT}/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export DOWNLOAD_MODELS_ON_START="${DOWNLOAD_MODELS_ON_START:-0}"
export REQUIRE_MODELS="${REQUIRE_MODELS:-1}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export PYTHONPATH="/opt/index-tts:${PYTHONPATH:-}"
export PATH="/opt/index-tts/.venv/bin:${PATH}"

W2V_CAND="${VOLUME_ROOT}/huggingface-cache/w2v-bert-2.0"
if [[ -f "${W2V_CAND}/config.json" ]]; then
  export W2V_BERT_PATH="${W2V_BERT_PATH:-$W2V_CAND}"
fi

log() { echo "[indextts-serverless] $*"; }

log "VOLUME_ROOT=$VOLUME_ROOT CHECKPOINTS_DIR=$CHECKPOINTS_DIR MODEL_VERSION=$MODEL_VERSION"

if [[ ! -d "$VOLUME_ROOT" ]]; then
  log "ERROR: VOLUME_ROOT 不存在: $VOLUME_ROOT"
  exit 1
fi

bash "${ROOT}/scripts/prepare_volume_models.sh"

# 预热：冷启动时加载权重，避免首请求超时
if [[ "${PRELOAD_MODEL:-1}" == "1" ]]; then
  log "preloading IndexTTS…"
  python - <<'PY' || { log "WARN: preload failed (will load on first job)"; true; }
from engine import get_tts, meta
get_tts()
print("[indextts] preload ok", meta())
PY
fi

log "starting RunPod handler"
exec python -u rp_handler.py
