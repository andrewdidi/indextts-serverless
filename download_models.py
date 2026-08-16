#!/usr/bin/env python3
"""下载 IndexTTS 权重到 Network Volume（snapshot_download）。

  python3 download_models.py --root /runpod-volume --version 2.5 --strict
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def repo_for_version(version: str) -> str:
    v = (version or "2.5").strip().lower()
    if v in {"2", "2.0", "tts2", "indextts-2"}:
        return "IndexTeam/IndexTTS-2"
    return "IndexTeam/IndexTTS-2.5"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("VOLUME_ROOT", "/runpod-volume"))
    p.add_argument("--version", default=os.environ.get("MODEL_VERSION", "2.5"))
    p.add_argument("--checkpoints-subdir", default="checkpoints")
    p.add_argument("--also-w2v", action="store_true", default=True)
    p.add_argument("--no-w2v", action="store_true")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    ckpt = root / args.checkpoints_subdir
    ckpt.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "huggingface-cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "huggingface-cache" / "hub"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: 需要 huggingface_hub", file=sys.stderr)
        return 1

    repo = repo_for_version(args.version)
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    print(f"[indextts] download {repo} → {ckpt}", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(ckpt),
        token=token,
        local_dir_use_symlinks=False,
    )

    if args.also_w2v and not args.no_w2v:
        w2v = root / "huggingface-cache" / "w2v-bert-2.0"
        if not (w2v / "config.json").is_file():
            print(f"[indextts] download facebook/w2v-bert-2.0 → {w2v}", flush=True)
            snapshot_download(
                repo_id="facebook/w2v-bert-2.0",
                local_dir=str(w2v),
                token=token,
                local_dir_use_symlinks=False,
            )
            os.environ["W2V_BERT_PATH"] = str(w2v)
        else:
            print(f"[indextts] w2v-bert already present: {w2v}", flush=True)
            os.environ["W2V_BERT_PATH"] = str(w2v)

    # 写出 marker 方便 prepare 脚本判断版本
    (ckpt / ".indextts_repo").write_text(repo + "\n", encoding="utf-8")

    if args.strict:
        from verify_models import verify

        ok, msg = verify(root, version=args.version)
        print(msg)
        return 0 if ok else 2

    print("[indextts] download done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
