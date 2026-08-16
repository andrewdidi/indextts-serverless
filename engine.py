#!/usr/bin/env python3
"""IndexTTS 引擎：懒加载单例，供 RunPod handler 调用。"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_TTS: Any = None
_META: dict[str, Any] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def checkpoints_dir() -> Path:
    explicit = (os.environ.get("CHECKPOINTS_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("VOLUME_ROOT") or "/runpod-volume")
    return root / "checkpoints"


def model_version() -> str:
    return (os.environ.get("MODEL_VERSION") or "2.5").strip()


def get_tts() -> Any:
    global _TTS, _META
    if _TTS is not None:
        return _TTS
    with _LOCK:
        if _TTS is not None:
            return _TTS
        ckpt = checkpoints_dir()
        cfg = ckpt / "config.yaml"
        if not cfg.is_file():
            raise FileNotFoundError(f"缺少 config.yaml: {cfg}（请先 Volume 填盘）")

        # 可选本地 w2v-bert
        w2v = Path(os.environ.get("W2V_BERT_PATH") or "")
        if not w2v.is_dir():
            cand = Path(os.environ.get("VOLUME_ROOT") or "/runpod-volume") / "huggingface-cache" / "w2v-bert-2.0"
            if (cand / "config.json").is_file():
                os.environ["W2V_BERT_PATH"] = str(cand)

        ver = model_version()
        use_bf16 = _env_bool("USE_BF16", True)
        use_fp16 = _env_bool("USE_FP16", False)
        use_cuda_kernel = _env_bool("USE_CUDA_KERNEL", False)
        use_deepspeed = _env_bool("USE_DEEPSPEED", False)
        use_qwen_emo = _env_bool("USE_QWEN_EMO", True)

        if ver in {"2", "2.0", "tts2"}:
            from indextts.infer_v2 import IndexTTS2

            _TTS = IndexTTS2(
                cfg_path=str(cfg),
                model_dir=str(ckpt),
                use_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel,
                use_deepspeed=use_deepspeed,
            )
            _META = {"version": "2", "use_fp16": use_fp16}
        else:
            from indextts.infer_v2_5 import IndexTTS2

            kwargs: dict[str, Any] = {
                "cfg_path": str(cfg),
                "model_dir": str(ckpt),
                "use_bf16": use_bf16,
            }
            # 2.5：文本情感需要 use_qwen_emo
            try:
                _TTS = IndexTTS2(**kwargs, use_qwen_emo=use_qwen_emo)
            except TypeError:
                _TTS = IndexTTS2(**kwargs)
            _META = {"version": "2.5", "use_bf16": use_bf16, "use_qwen_emo": use_qwen_emo}

        _META["checkpoints"] = str(ckpt)
        print(f"[indextts] model loaded: {_META}", flush=True)
        return _TTS


def meta() -> dict[str, Any]:
    get_tts()
    return dict(_META)


def synthesize(
    *,
    text: str,
    spk_audio_path: str,
    output_path: str,
    lang: str = "ZH",
    emo_audio_path: str | None = None,
    emo_vector: list[float] | None = None,
    emo_alpha: float = 1.0,
    use_emo_text: bool = False,
    emo_text: str | None = None,
    duration_factor: float = 1.0,
    use_random: bool = False,
    verbose: bool = False,
) -> str:
    tts = get_tts()
    ver = model_version()
    kwargs: dict[str, Any] = {
        "spk_audio_prompt": spk_audio_path,
        "text": text,
        "output_path": output_path,
        "verbose": verbose,
    }
    if emo_audio_path:
        kwargs["emo_audio_prompt"] = emo_audio_path
        kwargs["emo_alpha"] = float(emo_alpha)
    if emo_vector is not None:
        kwargs["emo_vector"] = emo_vector
        kwargs["use_random"] = bool(use_random)
    if use_emo_text:
        kwargs["use_emo_text"] = True
        kwargs["emo_alpha"] = float(emo_alpha if emo_alpha is not None else 0.6)
        if emo_text:
            kwargs["emo_text"] = emo_text
        kwargs["use_random"] = bool(use_random)

    if ver not in {"2", "2.0", "tts2"}:
        kwargs["lang"] = (lang or "ZH").upper()
        if duration_factor and abs(float(duration_factor) - 1.0) > 1e-6:
            kwargs["duration_factor"] = float(duration_factor)

    tts.infer(**kwargs)
    return output_path
