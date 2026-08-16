#!/usr/bin/env python3
"""提交 IndexTTS Serverless 请求并保存 wav。

  export RUNPOD_ENDPOINT_ID=xxx RUNPOD_API_KEY=rpa_xxx
  python3 send_request.py --request request.json --mode run --out-dir ./output
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def http_json(method: str, url: str, api_key: str, body: dict | None = None, timeout: int = 180) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:800]}") from e


def save_audio(output: dict, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    b64 = output.get("audio_base64")
    if not b64 and isinstance(output.get("audio"), str) and "base64," in output["audio"]:
        b64 = output["audio"].split("base64,", 1)[1]
    if not b64:
        return None
    path = out_dir / "latest_tts.wav"
    path.write_bytes(base64.b64decode(b64))
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
    p.add_argument("--mode", choices=["run", "runsync"], default="run")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--poll-s", type=float, default=3.0)
    p.add_argument("--max-wait-s", type=float, default=1800.0)
    args = p.parse_args()

    api_key = (args.api_key or "").strip()
    endpoint = (args.endpoint or "").strip()
    if not api_key or not endpoint:
        print("需要 RUNPOD_API_KEY 与 RUNPOD_ENDPOINT_ID（或 --api-key / --endpoint）")
        return 1

    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    base = f"https://api.runpod.ai/v2/{endpoint}"
    out_dir = Path(args.out_dir)
    t0 = time.time()

    if args.mode == "runsync":
        resp = http_json("POST", f"{base}/runsync", api_key, payload, timeout=int(args.max_wait_s))
    else:
        submitted = http_json("POST", f"{base}/run", api_key, payload, timeout=180)
        job_id = submitted.get("id")
        if not job_id:
            print(f"提交失败: {submitted}")
            return 1
        print(f"job={job_id}")
        deadline = time.time() + args.max_wait_s
        resp = {}
        while time.time() < deadline:
            resp = http_json("GET", f"{base}/status/{job_id}", api_key, timeout=90)
            st = resp.get("status")
            print(f"  status={st} elapsed={time.time()-t0:.0f}s")
            if st in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                break
            time.sleep(args.poll_s)
        else:
            print("等待超时")
            return 1

    slim = json.loads(json.dumps(resp))
    out = slim.get("output") if isinstance(slim.get("output"), dict) else {}
    if isinstance(out, dict):
        if out.get("audio_base64") and len(out["audio_base64"]) > 200:
            out["audio_base64"] = f"<base64 len={len(out['audio_base64'])}>"
        if isinstance(out.get("audio"), str) and len(out["audio"]) > 200:
            out["audio"] = f"<data-uri len={len(out['audio'])}>"
    (out_dir / "last_result.json").write_text(
        json.dumps({"result": slim, "elapsed_s": round(time.time() - t0, 1)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status = resp.get("status")
    output = resp.get("output") if isinstance(resp.get("output"), dict) else {}
    if status and status != "COMPLETED":
        print(f"失败: {status} {resp.get('error') or output.get('error')}")
        return 1
    if output.get("error") or output.get("ok") is False:
        print(f"失败: {output.get('error')}")
        return 1

    saved = save_audio(output, out_dir)
    if not saved:
        print("未返回音频，见 output/last_result.json")
        return 1
    print(f"OK → {saved} ({saved.stat().st_size} bytes)  elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
