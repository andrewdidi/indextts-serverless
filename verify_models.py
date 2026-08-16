#!/usr/bin/env python3
"""校验 Volume 上 IndexTTS checkpoints 是否齐全。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _size_ok(path: Path, expected: int | None, tol: float = 0.02) -> bool:
    if not path.is_file():
        return False
    if expected is None:
        return path.stat().st_size > 0
    actual = path.stat().st_size
    lo = int(expected * (1 - tol))
    hi = int(expected * (1 + tol))
    return lo <= actual <= hi


def required_files(version: str) -> list[tuple[str, int | None]]:
    v = (version or "2.5").strip().lower()
    if v in {"2", "2.0", "tts2", "indextts-2"}:
        return [
            ("config.yaml", None),
            ("gpt.pth", 3484663079),
            ("s2mel.pth", 1202198223),
            ("bpe.model", 475997),
            ("feat1.pt", 57170),
            ("feat2.pt", 374866),
            ("wav2vec2bert_stats.pt", 9343),
        ]
    return [
        ("config.yaml", None),
        ("gpt.pth", 3259599833),
        ("s2mel.pth", 414908601),
        ("codec.pth", 607290935),
        ("feat1.pt", 57170),
        ("feat2.pt", 374866),
        ("wav2vec2bert_stats.pt", 9343),
        ("multilingual_zh_ja_yue_char_del.tiktoken", 907395),
    ]


def verify(root: Path, *, version: str = "2.5", checkpoints_subdir: str = "checkpoints") -> tuple[bool, str]:
    ckpt = root / checkpoints_subdir
    missing: list[str] = []
    bad: list[str] = []
    for name, expect in required_files(version):
        path = ckpt / name
        if not path.is_file():
            missing.append(name)
        elif not _size_ok(path, expect):
            bad.append(f"{name} size={path.stat().st_size} expect≈{expect}")
    emo = ckpt / "qwen0.6bemo4-merge"
    if not emo.is_dir() or not any(emo.iterdir()):
        missing.append("qwen0.6bemo4-merge/")
    if missing or bad:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if bad:
            parts.append("bad_size: " + "; ".join(bad))
        return False, "[indextts] VERIFY FAIL — " + " | ".join(parts)
    return True, f"[indextts] VERIFY OK — {ckpt} (version={version})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("VOLUME_ROOT", "/runpod-volume"))
    p.add_argument("--version", default=os.environ.get("MODEL_VERSION", "2.5"))
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ok, msg = verify(Path(args.root), version=args.version)
    if args.json:
        print(json.dumps({"ok": ok, "message": msg}, ensure_ascii=False))
    else:
        print(msg)
    if args.strict and not ok:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
