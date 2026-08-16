#!/usr/bin/env bash
# 校验 / 按需下载 Volume 上的 IndexTTS checkpoints
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
MODEL_VERSION="${MODEL_VERSION:-2.5}"
DOWNLOAD_MODELS_ON_START="${DOWNLOAD_MODELS_ON_START:-0}"
REQUIRE_MODELS="${REQUIRE_MODELS:-1}"
ALLOW_MISSING_MODELS="${ALLOW_MISSING_MODELS:-0}"

export VOLUME_ROOT MODEL_VERSION
export PATH="/opt/index-tts/.venv/bin:${PATH}"
export PYTHONPATH="/opt/index-tts:${PYTHONPATH:-}"

log() { echo "[indextts] $*"; }

if python "${ROOT}/verify_models.py" --root "$VOLUME_ROOT" --version "$MODEL_VERSION"; then
  log "Volume 模型已齐全 → 跳过下载"
  exit 0
fi

if [[ "$DOWNLOAD_MODELS_ON_START" == "1" || "$DOWNLOAD_MODELS_ON_START" == "true" ]]; then
  log "模型缺失，开始下载（DOWNLOAD_MODELS_ON_START=1）…"
  python "${ROOT}/download_models.py" --root "$VOLUME_ROOT" --version "$MODEL_VERSION" --strict
  exit $?
fi

if [[ "$ALLOW_MISSING_MODELS" == "1" ]]; then
  log "WARN: 模型不齐但 ALLOW_MISSING_MODELS=1，继续启动"
  exit 0
fi

if [[ "$REQUIRE_MODELS" == "1" || "$REQUIRE_MODELS" == "true" ]]; then
  log "ERROR: 模型不齐。请用临时 Pod 执行: bash /app/scripts/pod_fill_volume.sh"
  log "  或设 DOWNLOAD_MODELS_ON_START=1（冷启动会拉全量，很慢）"
  exit 1
fi

log "WARN: REQUIRE_MODELS=0，跳过校验"
exit 0
