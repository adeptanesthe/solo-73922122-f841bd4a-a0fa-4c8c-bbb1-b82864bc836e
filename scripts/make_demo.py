"""Generate demo images and a ready-to-send recognize request.

Writes ``demo/line.png``, ``demo/glyph_<label>.png`` (11 samples),
``demo/spec.json`` and ``demo/request.sh``.  The demo line renders ``12.50``
with two fused glyph boundaries and two flipped pixels, so recognition has a
small, visible mismatch count.

Usage:  python scripts/make_demo.py [outdir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo_font import (  # noqa: E402
    DEMO_TEXT,
    all_glyph_labels,
    demo_line_png,
    demo_spec,
    glyph_png,
)


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "demo")
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "line.png").write_bytes(demo_line_png())
    for label in all_glyph_labels():
        fname = "glyph_dot.png" if label == "." else f"glyph_{label}.png"
        (outdir / fname).write_bytes(glyph_png(label))

    spec = demo_spec()
    (outdir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    cmd = "curl -sS -X POST http://localhost:8000/recognize \\\n"
    cmd += f"  -F 'line=@{outdir}/line.png;type=image/png' \\\n"
    for label in spec["glyph_labels"]:
        fname = "glyph_dot.png" if label == "." else f"glyph_{label}.png"
        cmd += f"  -F 'glyphs=@{outdir / fname};type=image/png' \\\n"
    cmd += f"  -F 'spec=<{outdir / 'spec.json'}'"
    (outdir / "request.sh").write_text(cmd + "\n", encoding="utf-8")

    print(f"wrote line.png (reading {DEMO_TEXT}), {len(spec['glyph_labels'])} glyphs,")
    print(f"spec.json and request.sh to {outdir}/")
    print("start the service with:  docker compose up --build")
    print(f"then run:                bash {outdir}/request.sh")
    print("interactive docs:        http://localhost:8000/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
