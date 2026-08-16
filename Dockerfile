# IndexTTS Serverless · slim 镜像（模型在 Network Volume）
# 上游: https://github.com/index-tts/index-tts  ·  默认权重 IndexTTS-2.5
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    INDEX_TTS_ROOT=/opt/index-tts \
    VOLUME_ROOT=/runpod-volume \
    MODEL_VERSION=2.5 \
    CHECKPOINTS_DIR=/runpod-volume/checkpoints \
    HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DOWNLOAD_MODELS_ON_START=0 \
    REQUIRE_MODELS=1 \
    USE_BF16=1 \
    USE_FP16=0 \
    USE_CUDA_KERNEL=0 \
    USE_DEEPSPEED=0

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl git git-lfs ffmpeg build-essential \
      python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && git lfs install \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt/index-tts
RUN git clone --depth 1 https://github.com/index-tts/index-tts.git /opt/index-tts \
 && uv sync --python 3.11 \
 && /opt/index-tts/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ENV PATH="/opt/index-tts/.venv/bin:/root/.local/bin:${PATH}" \
    PYTHONPATH="/opt/index-tts" \
    VIRTUAL_ENV=/opt/index-tts/.venv

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# uv 创建的 venv 默认无 pip，用 uv pip 往同一环境装 serverless 依赖
RUN uv pip install --python /opt/index-tts/.venv/bin/python --no-cache -r /app/requirements.txt \
 && /opt/index-tts/.venv/bin/python -c "import runpod, huggingface_hub; print('serverless deps ok')"

COPY handler.py engine.py input_normalize.py rp_handler.py /app/
COPY download_models.py verify_models.py models_manifest.json /app/
COPY scripts/prepare_volume_models.sh scripts/download_to_volume.sh scripts/pod_fill_volume.sh /app/scripts/
COPY start_worker.sh /app/start_worker.sh
COPY test_input.json /app/test_input.json

RUN chmod +x /app/start_worker.sh /app/scripts/*.sh /app/download_models.py /app/verify_models.py \
 && mkdir -p /app/tmp /app/output

WORKDIR /app
ENTRYPOINT ["/app/start_worker.sh"]
