"""Builds docs/demo.gif and docs/demo-screenshot.png from the frames in e2e/demo-frames/.

Run: python3 scripts/make_demo_gif.py   (after: cd e2e && node capture-demo.mjs)
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRAMES = sorted((ROOT / "e2e" / "demo-frames").glob("*.png"))
WIDTH = 760
# Show the page first, then each typed question and its answer a little longer.
HOLD_MS = {"home": 1800, "typed": 1400, "answer": 3200}


def load(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)


def main():
    if not FRAMES:
        raise SystemExit("No frames found. Run: cd e2e && node capture-demo.mjs")
    images = [load(p) for p in FRAMES]
    durations = [next(ms for key, ms in HOLD_MS.items() if key in p.name) for p in FRAMES]
    palette = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in images]
    out = ROOT / "docs" / "demo.gif"
    out.parent.mkdir(exist_ok=True)
    palette[0].save(out, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True)
    images[-1].save(ROOT / "docs" / "demo-screenshot.png", optimize=True)
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.0f} KB, {len(images)} frames")


if __name__ == "__main__":
    main()
