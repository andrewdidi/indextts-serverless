#!/usr/bin/env python3
"""IndexTTS 极速推理参数（对齐 センセイAgent / sensei_subtitle.core.indextts_turbo）。

CUDA Serverless 默认开启：beams=1、动态 max_mel、短分段。
BigVGAN 自定义 CUDA kernel 需 nvcc（devel 镜像）；slim runtime 默认关闭。
可通过环境变量 TURBO=0 关闭，或请求体 turbo{} 覆盖单项。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class IndexTurboConfig:
    enabled: bool = True
    use_fp16: bool = True
    use_cuda_kernel: bool = False
    use_accel: bool = False
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: int = 5
    max_mel_tokens: int = 480
    max_mel_tokens_floor: int = 96
    mel_tokens_per_char: int = 12
    dynamic_mel_tokens: bool = True
    max_text_tokens_per_segment: int = 64
    quick_streaming_tokens: int = 12
    interval_silence: int = 0
    repetition_penalty: float = 6.0
    length_penalty: float = 0.0
    diffusion_steps: int = 25
    inference_cfg_rate: float = 0.7


def _env_bool(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_turbo_from_env() -> IndexTurboConfig:
    """Serverless 默认开启 turbo（与 Sensei CUDA 极速一致）。"""
    return IndexTurboConfig(
        enabled=_env_bool("TURBO", True),
        use_fp16=_env_bool("USE_FP16", True),
        use_cuda_kernel=_env_bool("USE_CUDA_KERNEL", False),
        use_accel=_env_bool("USE_ACCEL", False),
        num_beams=_env_int("TURBO_NUM_BEAMS", 1, minimum=1),
        do_sample=_env_bool("TURBO_DO_SAMPLE", False),
        temperature=_env_float("TURBO_TEMPERATURE", 0.6),
        top_p=_env_float("TURBO_TOP_P", 0.6),
        top_k=_env_int("TURBO_TOP_K", 5, minimum=1),
        max_mel_tokens=_env_int("TURBO_MAX_MEL_TOKENS", 480, minimum=64),
        max_mel_tokens_floor=_env_int("TURBO_MAX_MEL_FLOOR", 96, minimum=48),
        mel_tokens_per_char=_env_int("TURBO_MEL_PER_CHAR", 12, minimum=8),
        dynamic_mel_tokens=_env_bool("TURBO_DYNAMIC_MEL", True),
        max_text_tokens_per_segment=_env_int("TURBO_MAX_TEXT_SEG", 64, minimum=8),
        quick_streaming_tokens=_env_int("TURBO_QUICK_STREAM", 12, minimum=0),
        interval_silence=_env_int("TURBO_INTERVAL_SILENCE", 0, minimum=0),
        repetition_penalty=_env_float("TURBO_REPETITION_PENALTY", 6.0),
        length_penalty=_env_float("TURBO_LENGTH_PENALTY", 0.0),
        diffusion_steps=_env_int("TURBO_DIFFUSION_STEPS", 25, minimum=4),
        inference_cfg_rate=_env_float("TURBO_INFERENCE_CFG_RATE", 0.7),
    )


def merge_turbo(base: IndexTurboConfig, overrides: dict[str, Any] | None) -> IndexTurboConfig:
    if not overrides:
        return base
    data = asdict(base)
    for key, val in overrides.items():
        if key not in data or val is None:
            continue
        if key == "enabled":
            data[key] = bool(val)
        elif key in {
            "use_fp16",
            "use_cuda_kernel",
            "use_accel",
            "do_sample",
            "dynamic_mel_tokens",
        }:
            data[key] = bool(val)
        elif key in {
            "temperature",
            "top_p",
            "repetition_penalty",
            "length_penalty",
            "inference_cfg_rate",
        }:
            data[key] = float(val)
        else:
            try:
                data[key] = int(val)
            except (TypeError, ValueError):
                continue
    return IndexTurboConfig(**data)


def dynamic_max_mel_tokens(text: str, turbo: IndexTurboConfig) -> int:
    n = len((text or "").strip())
    if n <= 0:
        return turbo.max_mel_tokens_floor
    estimated = n * turbo.mel_tokens_per_char + 24
    return max(turbo.max_mel_tokens_floor, min(turbo.max_mel_tokens, estimated))


def engine_init_kwargs(turbo: IndexTurboConfig, *, cuda: bool = True) -> dict[str, Any]:
    if not turbo.enabled:
        return {}
    return {
        "use_fp16": bool(turbo.use_fp16 and cuda),
        "use_cuda_kernel": bool(turbo.use_cuda_kernel and cuda),
        "use_accel": bool(turbo.use_accel and cuda),
    }


def synthesize_kwargs(turbo: IndexTurboConfig, *, text: str = "") -> dict[str, Any]:
    """
    传给 IndexTTS2.infer(**kwargs)。
    infer() 用 more_segment_before 位置映射到 quick_streaming_tokens，勿重复传后者。

    注意：官方 index-tts 2.5 的 infer 未 pop diffusion_steps / inference_cfg_rate，
    会漏进 GPT.generate 直接报错；故默认不传（扩散仍用上游默认 25 / 0.7）。
    若镜像内已打 Sensei 式 pop 补丁，可设 TURBO_PASS_DIFFUSION=1。
    """
    if not turbo.enabled:
        return {}
    max_mel = turbo.max_mel_tokens
    if turbo.dynamic_mel_tokens and (text or "").strip():
        max_mel = dynamic_max_mel_tokens(text, turbo)
    kw: dict[str, Any] = {
        "num_beams": turbo.num_beams,
        "do_sample": turbo.do_sample,
        "temperature": turbo.temperature,
        "top_p": turbo.top_p,
        "top_k": turbo.top_k,
        "max_mel_tokens": max_mel,
        "repetition_penalty": turbo.repetition_penalty,
        "length_penalty": turbo.length_penalty,
        "max_text_tokens_per_segment": turbo.max_text_tokens_per_segment,
        "interval_silence": turbo.interval_silence,
        "more_segment_before": turbo.quick_streaming_tokens,
    }
    if _env_bool("TURBO_PASS_DIFFUSION", False):
        kw["diffusion_steps"] = turbo.diffusion_steps
        kw["inference_cfg_rate"] = turbo.inference_cfg_rate
    return kw



def summary(turbo: IndexTurboConfig) -> dict[str, Any]:
    if not turbo.enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "num_beams": turbo.num_beams,
        "max_mel_tokens": turbo.max_mel_tokens,
        "dynamic_mel_tokens": turbo.dynamic_mel_tokens,
        "max_text_tokens_per_segment": turbo.max_text_tokens_per_segment,
        "quick_streaming_tokens": turbo.quick_streaming_tokens,
        "diffusion_steps": turbo.diffusion_steps,
        "interval_silence": turbo.interval_silence,
        "use_fp16": turbo.use_fp16,
        "use_cuda_kernel": turbo.use_cuda_kernel,
    }
