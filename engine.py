#!/usr/bin/env python3
"""IndexTTS 引擎：懒加载单例 + Sensei 对齐的 turbo 极速推理。"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from turbo import (
    IndexTurboConfig,
    engine_init_kwargs,
    load_turbo_from_env,
    merge_turbo,
    summary as turbo_summary,
    synthesize_kwargs,
)

_LOCK = threading.Lock()
_TTS: Any = None
_META: dict[str, Any] = {}
_TURBO: IndexTurboConfig | None = None


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


def spk_cache_dir() -> Path:
    explicit = (os.environ.get("SPK_CACHE_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("VOLUME_ROOT") or "/tmp")
    # Volume 可持久化跨 worker；无 Volume 时落 /tmp
    if root.is_dir() and str(root) not in {"/", ""}:
        return root / "indextts_spk_cache"
    return Path("/tmp/indextts_spk_cache")


def stable_spk_path(src: str | Path) -> Path:
    """
    按参考音频内容哈希落盘固定路径，触发 IndexTTS 内置 spk/emo 条件缓存。
    同一音色多次请求不会重复提取 embedding（Sensei 同会话复用路径的等价做法）。
    """
    src_path = Path(src)
    raw = src_path.read_bytes()
    digest = hashlib.sha1(raw).hexdigest()[:20]
    suf = src_path.suffix.lower() if src_path.suffix else ".wav"
    if suf not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}:
        suf = ".wav"
    dest = spk_cache_dir() / f"{digest}{suf}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.stat().st_size != len(raw):
        tmp = dest.with_suffix(dest.suffix + ".partial")
        tmp.write_bytes(raw)
        tmp.replace(dest)
    return dest


def get_turbo() -> IndexTurboConfig:
    global _TURBO
    if _TURBO is None:
        _TURBO = load_turbo_from_env()
    return _TURBO


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

        w2v = Path(os.environ.get("W2V_BERT_PATH") or "")
        if not w2v.is_dir():
            cand = Path(os.environ.get("VOLUME_ROOT") or "/runpod-volume") / "huggingface-cache" / "w2v-bert-2.0"
            if (cand / "config.json").is_file():
                os.environ["W2V_BERT_PATH"] = str(cand)

        ver = model_version()
        turbo = get_turbo()
        use_bf16 = _env_bool("USE_BF16", True)
        use_deepspeed = _env_bool("USE_DEEPSPEED", False)
        # 文本情感需 Qwen；默认关以加快加载（emo_vector / emo_audio 仍可用）
        use_qwen_emo = _env_bool("USE_QWEN_EMO", False)

        init_turbo = engine_init_kwargs(turbo, cuda=True)
        use_fp16 = bool(init_turbo.get("use_fp16", False)) or _env_bool("USE_FP16", False)
        use_cuda_kernel = bool(init_turbo.get("use_cuda_kernel", False)) or _env_bool(
            "USE_CUDA_KERNEL", False
        )
        use_accel = bool(init_turbo.get("use_accel", False))

        if ver in {"2", "2.0", "tts2"}:
            from indextts.infer_v2 import IndexTTS2

            kwargs: dict[str, Any] = {
                "cfg_path": str(cfg),
                "model_dir": str(ckpt),
                "use_fp16": use_fp16,
                "use_cuda_kernel": use_cuda_kernel,
                "use_deepspeed": use_deepspeed,
            }
            if use_accel:
                kwargs["use_accel"] = True
            try:
                _TTS = IndexTTS2(**kwargs)
            except TypeError:
                kwargs.pop("use_accel", None)
                _TTS = IndexTTS2(**kwargs)
            _META = {
                "version": "2",
                "use_fp16": use_fp16,
                "use_cuda_kernel": use_cuda_kernel,
                "use_accel": use_accel,
            }
        else:
            from indextts.infer_v2_5 import IndexTTS2

            # 必须显式传 use_cuda_kernel=False：上游默认 (None → True on CUDA)，
            # slim runtime 无 nvcc，不传会每次冷启动尝试编译并打印 Falling back。
            kwargs = {
                "cfg_path": str(cfg),
                "model_dir": str(ckpt),
                "use_bf16": use_bf16,
                "use_fp16": use_fp16,
                "use_cuda_kernel": use_cuda_kernel,
                "use_deepspeed": use_deepspeed,
                "use_qwen_emo": use_qwen_emo,
            }
            if use_accel:
                kwargs["use_accel"] = True
            try:
                _TTS = IndexTTS2(**kwargs)
            except TypeError:
                # 逐级剥离可选参数，兼容不同上游签名
                for drop in ("use_accel", "use_deepspeed", "use_fp16", "use_qwen_emo"):
                    kwargs.pop(drop, None)
                    try:
                        _TTS = IndexTTS2(**kwargs)
                        break
                    except TypeError:
                        continue
                else:
                    _TTS = IndexTTS2(
                        cfg_path=str(cfg),
                        model_dir=str(ckpt),
                        use_bf16=use_bf16,
                        use_cuda_kernel=use_cuda_kernel,
                    )
            _META = {
                "version": "2.5",
                "use_bf16": use_bf16,
                "use_fp16": use_fp16,
                "use_cuda_kernel": use_cuda_kernel,
                "use_qwen_emo": use_qwen_emo,
            }

        _META["checkpoints"] = str(ckpt)
        _META["turbo"] = turbo_summary(turbo)
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
    turbo_overrides: dict[str, Any] | None = None,
) -> str:
    tts = get_tts()
    ver = model_version()
    turbo = merge_turbo(get_turbo(), turbo_overrides)

    if use_emo_text and not _META.get("use_qwen_emo"):
        raise ValueError(
            "use_emo_text 需要环境变量 USE_QWEN_EMO=1 并 Redeploy 预热；"
            "或改用 emo_vector / emo_audio"
        )

    # 稳定路径 → 命中 IndexTTS 内置 spk/emo 条件缓存
    spk_stable = str(stable_spk_path(spk_audio_path))
    emo_stable = str(stable_spk_path(emo_audio_path)) if emo_audio_path else None

    kwargs: dict[str, Any] = {
        "spk_audio_prompt": spk_stable,
        "text": text,
        "output_path": output_path,
        "verbose": verbose,
    }
    if emo_stable:
        kwargs["emo_audio_prompt"] = emo_stable
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
    else:
        # v2 用 text_lang
        lang_l = (lang or "zh").strip().lower()
        if lang_l in {"zh", "cn", "chinese", "zh-cn"}:
            kwargs["text_lang"] = "zh"
        elif lang_l in {"en", "english"}:
            kwargs["text_lang"] = "en"
        elif lang_l in {"ja", "jp", "japanese"}:
            kwargs["text_lang"] = "ja"

    # Sensei turbo：beams=1 / 动态 mel / 短分段 …
    kwargs.update(synthesize_kwargs(turbo, text=text))
    # 双保险：绝不把扩散参数漏进 GPT.generate（上游 2.5 常见坑）
    if not _env_bool("TURBO_PASS_DIFFUSION", False):
        kwargs.pop("diffusion_steps", None)
        kwargs.pop("inference_cfg_rate", None)

    try:
        tts.infer(**kwargs)
    except (TypeError, ValueError) as e:
        msg = str(e)
        # 旧版 / 未 pop 的 kwargs：剥离后再试
        drop_candidates = (
            "diffusion_steps",
            "inference_cfg_rate",
            "more_segment_before",
            "max_text_tokens_per_segment",
            "interval_silence",
            "num_beams",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "max_mel_tokens",
            "repetition_penalty",
            "length_penalty",
            "text_lang",
            "lang",
            "duration_factor",
        )
        dropped = False
        for k in drop_candidates:
            if k in kwargs and (k in msg or "model_kwargs" in msg or "not used" in msg):
                kwargs.pop(k, None)
                dropped = True
        if not dropped:
            for k in ("diffusion_steps", "inference_cfg_rate", "more_segment_before"):
                if k in kwargs:
                    kwargs.pop(k, None)
                    dropped = True
        if not dropped:
            raise
        print(f"[indextts] retry infer after stripping kwargs: {e}", flush=True)
        tts.infer(**kwargs)

    _META["turbo_last"] = turbo_summary(turbo)
    if turbo.enabled and turbo.dynamic_mel_tokens:
        from turbo import dynamic_max_mel_tokens

        _META["turbo_last"]["max_mel_tokens_effective"] = dynamic_max_mel_tokens(text, turbo)
    return output_path
