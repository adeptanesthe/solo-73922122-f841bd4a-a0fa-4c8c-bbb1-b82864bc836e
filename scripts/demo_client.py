"""In-process CLI: build demo tiles, POST them to a running service, print JSON.

This is the no-Docker fallback for the demo:

    uvicorn app.main:app --port 8000 &
    python scripts/demo_client.py --base-url http://localhost:8000

It exercises the real HTTP surface (multipart upload + JSON spec), not the
Python functions directly, and exits non-zero if the service does not answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_data import OFFSETS, demo_tiles, png_bytes, window  # noqa: E402


def build_multipart(tiles: list[bytes], spec: str) -> tuple[bytes, str]:
    boundary = "----tilealigndemo"
    parts: list[bytes] = []
    for idx, data in enumerate(tiles):
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="images"; '
                f'filename="tile{idx}.png"\r\nContent-Type: image/png\r\n\r\n'
            ).encode()
            + data
            + b"\r\n"
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="spec"\r\n\r\n{spec}\r\n'
        ).encode()
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    tiles = [png_bytes(t) for t in demo_tiles()]
    spec = json.dumps(
        {
            "pairs": [
                {"a": a, "b": b, "window": window(off, radius=2)}
                for (a, b), off in OFFSETS.items()
            ],
            "min_overlap_pixels": 20000,
            "max_mean_abs_diff": 3.0,
        }
    )
    body, content_type = build_multipart(tiles, spec)
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/align",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - demo tool
        print(f"request failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") == "aligned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
