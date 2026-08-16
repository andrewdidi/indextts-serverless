#!/usr/bin/env python3
"""规范化 RunPod input → 本地临时音频路径 + 推理参数。"""

from __future__ import annotations

import base64
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


_DATA_URI_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.I | re.S)


def _decode_audio_blob(raw: str, dest: Path) -> Path:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("音频为空")
    m = _DATA_URI_RE.match(raw)
    if m:
        payload = m.group(3)
        is_b64 = bool(m.group(2))
        if is_b64:
            dest.write_bytes(base64.b64decode(payload))
        else:
            dest.write_bytes(urllib.parse.unquote_to_bytes(payload))
        return dest
    # 纯 base64
    try:
        dest.write_bytes(base64.b64decode(raw, validate=False))
        return dest
    except Exception as e:
        raise ValueError(f"无法解码音频 base64: {e}") from e


def _download(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "indextts-serverless/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return dest


def _write_audio_field(value: Any, dest: Path) -> Path:
    if value is None:
        raise ValueError("缺少参考音频")
    if isinstance(value, (bytes, bytearray)):
        dest.write_bytes(bytes(value))
        return dest
    if not isinstance(value, str):
        raise ValueError("音频字段须为 base64 / data URI / URL 字符串")
    s = value.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return _download(s, dest)
    return _decode_audio_blob(s, dest)


def normalize_input(job: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    text = (job.get("text") or job.get("prompt") or "").strip()
    if not text:
        raise ValueError("缺少 text")

    spk = job.get("spk_audio") or job.get("speaker_audio") or job.get("voice") or job.get("reference_audio")
    spk_url = job.get("spk_audio_url") or job.get("speaker_audio_url") or job.get("voice_url")
    if spk is None and spk_url:
        spk = spk_url
    if spk is None:
        raise ValueError("缺少 spk_audio（参考音色 wav/mp3 base64 或 URL）")

    spk_path = work_dir / "spk_prompt.wav"
    _write_audio_field(spk, spk_path)

    emo_path = None
    emo = job.get("emo_audio") or job.get("emotion_audio")
    emo_url = job.get("emo_audio_url") or job.get("emotion_audio_url")
    if emo is None and emo_url:
        emo = emo_url
    if emo:
        emo_path = str(work_dir / "emo_prompt.wav")
        _write_audio_field(emo, Path(emo_path))

    emo_vector = job.get("emo_vector")
    if emo_vector is not None:
        if not isinstance(emo_vector, (list, tuple)) or len(emo_vector) != 8:
            raise ValueError("emo_vector 须为长度 8 的数组 [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]")
        emo_vector = [float(x) for x in emo_vector]

    lang = (job.get("lang") or job.get("language") or "ZH").strip().upper()
    if lang in {"CN", "CHINESE", "ZH-CN"}:
        lang = "ZH"
    elif lang in {"ENGLISH"}:
        lang = "EN"

    out = {
        "text": text,
        "spk_audio_path": str(spk_path),
        "emo_audio_path": emo_path,
        "emo_vector": emo_vector,
        "emo_alpha": float(job.get("emo_alpha", 1.0)),
        "use_emo_text": bool(job.get("use_emo_text", False)),
        "emo_text": (job.get("emo_text") or None),
        "duration_factor": float(job.get("duration_factor", 1.0)),
        "use_random": bool(job.get("use_random", False)),
        "lang": lang,
        "verbose": bool(job.get("verbose", False)),
    }
    return out
