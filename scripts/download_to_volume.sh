#!/usr/bin/env bash
# 强制下载到 Volume（填盘 / 补缺）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
export MODEL_VERSION="${MODEL_VERSION:-2.5}"
export PATH="/opt/index-tts/.venv/bin:${PATH}"
export PYTHONPATH="/opt/index-tts:${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
echo "[indextts] download_to_volume VOLUME_ROOT=$VOLUME_ROOT version=$MODEL_VERSION"
python "${ROOT}/download_models.py" --root "$VOLUME_ROOT" --version "$MODEL_VERSION" --strict
