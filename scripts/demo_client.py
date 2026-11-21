"""In-process CLI: build demo images, POST them to a running service, print JSON.

    uvicorn app.main:app --port 8000 &
    python scripts/demo_client.py --base-url http://localhost:8000

It exercises the real HTTP surface (multipart upload + JSON spec) and exits
non-zero unless the service answers with status ``entry_ready``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo_font import all_glyph_labels, demo_line_png, demo_spec, glyph_png  # noqa: E402


def build_multipart(line: bytes, glyphs: list[tuple[str, bytes]], spec: str):
    boundary = "----dimensiondemo"
    parts: list[bytes] = [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="line"; '
            'filename="line.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        + line
        + b"\r\n"
    ]
    for name, data in glyphs:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="glyphs"; '
                f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n'
            ).encode()
            + data
            + b"\r\n"
        )
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"spec\"\r\n\r\n{spec}\r\n").encode()
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    labels = all_glyph_labels()
    glyphs = [
        ("glyph_dot.png" if label == "." else f"glyph_{label}.png", glyph_png(label))
        for label in labels
    ]
    body, content_type = build_multipart(demo_line_png(), glyphs, json.dumps(demo_spec()))
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/recognize",
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
    return 0 if payload.get("status") == "entry_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
