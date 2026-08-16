# IndexTTS · 零出错部署清单

镜像：`ghcr.io/andrewdidi/indextts-serverless:latest`  
上游：[index-tts](https://github.com/index-tts/index-tts) · 默认权重 [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)

## 绿灯标准

1. Network Volume ≥ **30GB**，挂载 `/runpod-volume`
2. `verify_models.py --strict` 通过（`checkpoints/gpt.pth` 等齐全）
3. GHCR 包 **Public**（或配 Registry Auth）
4. Endpoint 环境变量：
   ```text
   VOLUME_ROOT=/runpod-volume
   DOWNLOAD_MODELS_ON_START=0
   REQUIRE_MODELS=1
   MODEL_VERSION=2.5
   ```
5. 日志：`Volume 模型已齐全 → 跳过下载` / `model loaded`

## 步骤

### 1. 镜像

推送本仓库后 Actions 构建 GHCR；或本机构建：

```bash
docker build -t ghcr.io/andrewdidi/indextts-serverless:latest .
docker push ghcr.io/andrewdidi/indextts-serverless:latest
```

### 2. 临时 Pod 填盘

| 项 | 值 |
|----|-----|
| Image | `ghcr.io/andrewdidi/indextts-serverless:latest` |
| Volume | → `/runpod-volume` |
| Env | `VOLUME_ROOT=/runpod-volume` · `HF_TOKEN=hf_xxx` · `MODEL_VERSION=2.5` |
| Start Command | `bash /app/scripts/pod_fill_volume.sh` |

约 **5–8GB**（含 w2v-bert 缓存更多）。日志见 `PASSED` 后 Stop Pod。

### 3. Serverless

同一 Volume + 上方环境变量。GPU 建议 ≥16–24GB（BF16）。

### 4. 本地调用

```bash
python3 build_request.py \
  --spk ./voice.wav \
  --text "大家好，欢迎使用 IndexTTS。" \
  --lang ZH \
  --out request.json

export RUNPOD_ENDPOINT_ID=xxx RUNPOD_API_KEY=rpa_xxx
python3 send_request.py --request request.json --mode run --out-dir ./output
```

或双击 `选中执行_IndexTTS.command` / `Exe_UI/index_tts`。

## Volume 布局

```text
/runpod-volume/
  checkpoints/          # IndexTTS-2.5 权重
    config.yaml
    gpt.pth
    s2mel.pth
    codec.pth
    qwen0.6bemo4-merge/
    …
  huggingface-cache/
    w2v-bert-2.0/       # 可选预热
    hub/
```

## 输入约定

```json
{
  "input": {
    "text": "要说的话",
    "spk_audio": "data:audio/wav;base64,...",
    "lang": "ZH",
    "duration_factor": 1.0,
    "emo_audio": null,
    "emo_vector": null,
    "use_emo_text": false
  }
}
```

返回：`audio_base64` / `audio`（data URI）wav。

## 说明

- `MODEL_VERSION=2` 可切 IndexTTS-2（填盘时用同一变量）。
- 日常 Endpoint 务必 `DOWNLOAD_MODELS_ON_START=0`，避免冷启动拉模型超时。
