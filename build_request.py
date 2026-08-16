#!/usr/bin/env python3
"""构建 IndexTTS Serverless 请求 JSON。

  python3 build_request.py --spk ./voice.wav --text "大家好" --lang ZH --out request.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def encode_audio(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "wav"
    mime = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spk", required=True, help="参考音色音频")
    p.add_argument("--text", required=True)
    p.add_argument("--lang", default="ZH")
    p.add_argument("--emo", default=None, help="可选情绪参考音频")
    p.add_argument("--emo-alpha", type=float, default=1.0)
    p.add_argument("--duration-factor", type=float, default=1.0)
    p.add_argument("--use-emo-text", action="store_true")
    p.add_argument("--emo-text", default=None)
    p.add_argument("--out", default="request.json")
    args = p.parse_args()

    spk = Path(args.spk)
    if not spk.is_file():
        print(f"找不到参考音频: {spk}")
        return 1

    inp: dict = {
        "text": args.text,
        "spk_audio": encode_audio(spk),
        "lang": args.lang,
        "duration_factor": args.duration_factor,
    }
    if args.emo:
        emo = Path(args.emo)
        if not emo.is_file():
            print(f"找不到情绪音频: {emo}")
            return 1
        inp["emo_audio"] = encode_audio(emo)
        inp["emo_alpha"] = args.emo_alpha
    if args.use_emo_text:
        inp["use_emo_text"] = True
        inp["emo_alpha"] = args.emo_alpha if args.emo_alpha != 1.0 else 0.6
        if args.emo_text:
            inp["emo_text"] = args.emo_text

    payload = {
        "input": inp,
        "policy": {"executionTimeout": 1_800_000, "ttl": 3_600_000},
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
