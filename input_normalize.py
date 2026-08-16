#!/usr/bin/env python3
"""规范化 RunPod input → 本地临时音频路径 + 推理参数。"""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


_DATA_URI_RE = re.compile(r"^data:([^;,]+)?((?:;[^,]*)*),(.*)$", re.I | re.S)
_PLACEHOLDER_RE = re.compile(r"^(<|placeholder|paste |xxx|\.\.\.|null|none)", re.I)


def _b64decode_padded(s: str) -> bytes:
    """容错 base64：去空白、urlsafe、自动补 padding。"""
    t = (s or "").strip()
    if not t:
        raise ValueError("base64 为空")
    # 常见前缀噪音
    if t.lower().startswith("base64,"):
        t = t.split(",", 1)[1]
    t = "".join(t.split())  # 去换行/空格
    # urlsafe → 标准
    if "-" in t or "_" in t:
        t = t.replace("-", "+").replace("_", "/")
    # 去掉非法字符（保留 A-Za-z0-9+/=）
    t = re.sub(r"[^A-Za-z0-9+/=]", "", t)
    pad = (-len(t)) % 4
    if pad:
        t += "=" * pad
    try:
        return base64.b64decode(t, validate=False)
    except binascii.Error:
        # 再试：截断到 4 的倍数
        t2 = t.rstrip("=")
        t2 = t2[: len(t2) - (len(t2) % 4)]
        if not t2:
            raise
        return base64.b64decode(t2 + "=" * ((-len(t2)) % 4), validate=False)


def _looks_like_audio(blob: bytes) -> bool:
    if len(blob) < 12:
        return False
    if blob[:4] == b"RIFF" and blob[8:12] == b"WAVE":
        return True
    if blob[:3] == b"ID3" or blob[:2] == b"\xff\xfb" or blob[:2] == b"\xff\xf3":
        return True  # mp3
    if blob[:4] == b"fLaC" or blob[:4] == b"OggS":
        return True
    if blob[:4] == b"ftyp" or blob[4:8] == b"ftyp":
        return True  # m4a
    # 未知容器但足够大，仍接受（交给下游 ffmpeg/librosa）
    return len(blob) >= 256


def _decode_audio_blob(raw: str, dest: Path) -> Path:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("音频为空")
    if _PLACEHOLDER_RE.match(raw) or "build_request" in raw or "paste data:" in raw.lower():
        raise ValueError(
            "spk_audio 仍是占位符。请用真实 wav/mp3 的 base64 或 data URI："
            "python3 build_request.py --spk ./voice.wav --text '你好' --out request.json"
        )

    m = _DATA_URI_RE.match(raw)
    if m:
        meta = (m.group(2) or "").lower()
        payload = m.group(3)
        if "base64" in meta or _looks_like_b64(payload):
            blob = _b64decode_padded(payload)
        else:
            blob = urllib.parse.unquote_to_bytes(payload)
        if not _looks_like_audio(blob):
            # data URI 解码后不像音频时再试一次纯 base64
            try:
                alt = _b64decode_padded(payload)
                if _looks_like_audio(alt):
                    blob = alt
            except Exception:
                pass
        dest.write_bytes(blob)
        return dest

    # 纯 base64 / 偶发带 data: 但格式奇怪
    if raw.lower().startswith("data:") and "," in raw:
        payload = raw.split(",", 1)[1]
        blob = _b64decode_padded(payload)
        dest.write_bytes(blob)
        return dest

    try:
        blob = _b64decode_padded(raw)
    except Exception as e:
        raise ValueError(
            f"无法解码音频 base64: {e}。"
            "请确认 spk_audio 是完整 base64（勿截断），或使用 data:audio/wav;base64,..."
        ) from e
    if not _looks_like_audio(blob):
        raise ValueError(
            f"解码后不像音频文件（{len(blob)} bytes）。"
            "请用 build_request.py 重新编码，或传 spk_audio_url 指向可下载的 wav。"
        )
    dest.write_bytes(blob)
    return dest


def _looks_like_b64(s: str) -> bool:
    sample = "".join((s or "")[:80].split())
    if len(sample) < 16:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=_-]+", sample))


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
    # RunPod / 部分客户端可能包一层 dict
    if isinstance(value, dict):
        inner = value.get("data") or value.get("audio") or value.get("base64") or value.get("content")
        if inner is None:
            raise ValueError("音频 dict 缺少 data/audio/base64 字段")
        return _write_audio_field(inner, dest)
    if isinstance(value, list) and value and all(isinstance(x, int) for x in value[:32]):
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

    return {
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
