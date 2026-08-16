#!/usr/bin/env python3
"""简易 CLI：交互构建请求并提交 IndexTTS Serverless。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"


def load_cfg() -> dict:
    cfg = {"endpoint_id": "", "api_key": "", "lang": "ZH"}
    if CONFIG.is_file():
        try:
            cfg.update(json.loads(CONFIG.read_text(encoding="utf-8")))
        except Exception:
            pass
    cfg["api_key"] = cfg.get("api_key") or os.environ.get("RUNPOD_API_KEY", "")
    cfg["endpoint_id"] = cfg.get("endpoint_id") or os.environ.get("RUNPOD_ENDPOINT_ID", "")
    return cfg


def save_cfg(cfg: dict) -> None:
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    cfg = load_cfg()
    print("=== IndexTTS Serverless ===")
    print(f"endpoint: {cfg.get('endpoint_id') or '(空)'}")
    spk = input("参考音色路径 [.wav/.mp3]: ").strip()
    text = input("合成文本: ").strip()
    lang = input(f"语言 [{cfg.get('lang','ZH')}]: ").strip() or cfg.get("lang", "ZH")
    ep = input(f"Endpoint ID [{cfg.get('endpoint_id','')}]: ").strip() or cfg.get("endpoint_id", "")
    key = input("API Key（回车沿用已有）: ").strip() or cfg.get("api_key", "")
    if not spk or not text or not ep or not key:
        print("缺少必要项")
        return 1
    cfg.update({"endpoint_id": ep, "api_key": key, "lang": lang})
    save_cfg(cfg)

    req = ROOT / "output" / "request.json"
    req.parent.mkdir(parents=True, exist_ok=True)
    r1 = subprocess.run(
        [sys.executable, str(ROOT / "build_request.py"), "--spk", spk, "--text", text, "--lang", lang, "--out", str(req)],
        check=False,
    )
    if r1.returncode != 0:
        return r1.returncode
    env = os.environ.copy()
    env["RUNPOD_ENDPOINT_ID"] = ep
    env["RUNPOD_API_KEY"] = key
    return subprocess.call(
        [sys.executable, str(ROOT / "send_request.py"), "--request", str(req), "--mode", "run", "--out-dir", str(ROOT / "output")],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
