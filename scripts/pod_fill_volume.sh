#!/usr/bin/env bash
# 临时 Pod 专用：只补齐 Network Volume 模型，不启动 Serverless handler。
# RunPod Pod「Docker Start Command」:
#   bash /app/scripts/pod_fill_volume.sh
set -euo pipefail

export VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
export DOWNLOAD_MODELS_ON_START=1
export REQUIRE_MODELS=1
export MODEL_VERSION="${MODEL_VERSION:-2.5}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export PATH="/opt/index-tts/.venv/bin:${PATH}"
export PYTHONPATH="/opt/index-tts:${PYTHONPATH:-}"

echo "============================================================"
echo "[indextts] Pod 填盘开始（不启动 handler）"
echo "  VOLUME_ROOT=$VOLUME_ROOT  MODEL_VERSION=$MODEL_VERSION"
if [[ -n "${HF_TOKEN}" ]]; then echo "  HF_TOKEN: set"; else echo "  HF_TOKEN: empty"; fi
echo "============================================================"

if [[ ! -d "$VOLUME_ROOT" ]]; then
  echo "[indextts] ERROR: VOLUME_ROOT 不存在: $VOLUME_ROOT"
  echo "  请挂载 Network Volume → /runpod-volume（建议 ≥30GB）"
  exit 1
fi

df -h "$VOLUME_ROOT" || true
bash /app/scripts/download_to_volume.sh

echo ""
echo "============================================================"
echo "[indextts] ✅ Volume 模型已齐全（PASSED）"
echo "下一步:"
echo "  1. Stop 本临时 Pod"
echo "  2. Serverless 挂同一 Volume → /runpod-volume"
echo "  3. Env:"
echo "       VOLUME_ROOT=/runpod-volume"
echo "       DOWNLOAD_MODELS_ON_START=0"
echo "       REQUIRE_MODELS=1"
echo "       MODEL_VERSION=2.5"
echo "============================================================"

KEEP_ALIVE="${KEEP_ALIVE:-1}"
if [[ "$KEEP_ALIVE" == "1" ]]; then
  echo "[indextts] KEEP_ALIVE=1 → sleep infinity"
  exec sleep infinity
fi
