# Index-TTS Serverless

RunPod Serverless 部署包：**IndexTTS-2.5**（可切 2.0）零样本音色克隆 TTS。

- 上游代码：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- 权重：[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)
- 镜像：`ghcr.io/andrewdidi/indextts-serverless:latest`

## 快速路径

见 **[DEPLOY.md](./DEPLOY.md)**（Volume 填盘 → Serverless → 本地调用）。

本地：

```bash
./选中执行_IndexTTS.command
# 或
python3 run_ui.py
```

网页 UI：`Exe_UI/index_tts/选中执行_IndexTTS.command`

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `VOLUME_ROOT` | `/runpod-volume` | Network Volume |
| `CHECKPOINTS_DIR` | `$VOLUME_ROOT/checkpoints` | 权重目录 |
| `MODEL_VERSION` | `2.5` | `2.5` 或 `2` |
| `DOWNLOAD_MODELS_ON_START` | `0` | Serverless 日常勿开 |
| `REQUIRE_MODELS` | `1` | 缺模型 fail-fast |
| `USE_BF16` | `1` | 2.5 半精度 |
| `HF_TOKEN` | — | 填盘下载用 |

## License

IndexTTS 模型与代码遵循上游 Bilibili IndexTTS 许可；本仓库仅部署编排脚本。
