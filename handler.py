#!/usr/bin/env python3
"""IndexTTS RunPod handler：参考音色 + 文本 → wav（base64）。"""

from __future__ import annotations

import base64
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from engine import meta, synthesize
from input_normalize import normalize_input


def handler(event: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    job = event.get("input") if isinstance(event, dict) else None
    if job is None and isinstance(event, dict):
        job = event
    job = job if isinstance(job, dict) else {}

    work = Path(tempfile.mkdtemp(prefix="indextts_", dir=os.environ.get("TMPDIR") or "/tmp"))
    out_wav = work / "gen.wav"
    try:
        params = normalize_input(job, work)
        synthesize(
            text=params["text"],
            spk_audio_path=params["spk_audio_path"],
            output_path=str(out_wav),
            lang=params["lang"],
            emo_audio_path=params.get("emo_audio_path"),
            emo_vector=params.get("emo_vector"),
            emo_alpha=params.get("emo_alpha", 1.0),
            use_emo_text=params.get("use_emo_text", False),
            emo_text=params.get("emo_text"),
            duration_factor=params.get("duration_factor", 1.0),
            use_random=params.get("use_random", False),
            verbose=params.get("verbose", False),
            turbo_overrides=params.get("turbo_overrides"),
        )
        if not out_wav.is_file() or out_wav.stat().st_size < 44:
            raise RuntimeError("推理未生成有效 wav")
        blob = out_wav.read_bytes()
        b64 = base64.b64encode(blob).decode("ascii")
        elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "ok": True,
            "audio_base64": b64,
            "audio": f"data:audio/wav;base64,{b64}",
            "audio_format": "wav",
            "bytes": len(blob),
            "lang": params["lang"],
            "text": params["text"],
            "model": meta(),
            "elapsed_ms": elapsed,
            "_indextts_serverless": True,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-2000:],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "_indextts_serverless": True,
        }
    finally:
        try:
            for p in work.glob("*"):
                p.unlink(missing_ok=True)
            work.rmdir()
        except OSError:
            pass
