"""Generate a demo tile set and a ready-to-send align request.

Writes ``demo/tileN.png`` (four overlapping crops of one synthetic drawing that
form a 2x2 mosaic), ``demo/spec.json`` and ``demo/request.sh``.  The pair list
includes the redundant cycle pair (0,3), so a unique answer requires the cycle
to agree -- a good end-to-end check of the service.

Usage:  python scripts/make_demo.py [outdir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_data import OFFSETS, demo_tiles, png_bytes, window  # noqa: E402


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "demo")
    outdir.mkdir(parents=True, exist_ok=True)

    tiles = demo_tiles()
    for idx, tile in enumerate(tiles):
        (outdir / f"tile{idx}.png").write_bytes(png_bytes(tile))

    spec = {
        "pairs": [
            {"a": a, "b": b, "window": window(off, radius=2)} for (a, b), off in OFFSETS.items()
        ],
        "min_overlap_pixels": 20000,
        "max_mean_abs_diff": 3.0,
    }
    (outdir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    cmd = "curl -sS -X POST http://localhost:8000/align \\\n" + "".join(
        f"  -F 'images=@{outdir}/tile{idx}.png;type=image/png' \\\n"
        for idx in range(len(tiles))
    )
    cmd += f"  -F 'spec=<{outdir / 'spec.json'}'"
    (outdir / "request.sh").write_text(cmd + "\n", encoding="utf-8")

    print(f"wrote {len(tiles)} tiles, spec.json and request.sh to {outdir}/")
    print("start the service with:  docker compose up --build")
    print(f"then run:                bash {outdir}/request.sh")
    print("interactive docs:        http://localhost:8000/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
