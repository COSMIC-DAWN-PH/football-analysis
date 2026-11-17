"""Build labelled contact sheets of ball review crops for bulk review.

Every crop of a manifest is placed into a fixed-size grid cell with its item
id drawn as a caption, so the dual models (qwen3.7plus / gpt5.6luna) and the
human can review 20 crops per vision call / per glance. Individual crops stay
available in crops/ for zooming into uncertain cells.

Outputs:
  sheets/<src>/sheet_0001.png ...  grid images
  sheets/<src>/sheets.jsonl         {"sheet": "sheet_0001.png", "ids": [...]}
  sheets/<src>/sheet_of_id.jsonl    {"<item-id>": "sheet_0001.png"}

Usage:
  python tools/make_ball_sheets.py --manifest eval/ball_crops/demo4/candidates.jsonl \
      --out eval/ball_crops/demo4/sheets --cells 20 --cell 288
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

CAPTION_H = 18


def _load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _cell_image(crop_path: Path, cell: int) -> np.ndarray:
    img = cv2.imread(str(crop_path))
    if img is None:
        img = np.full((cell, cell, 3), 40, dtype=np.uint8)
    h, w = img.shape[:2]
    inner = cell - CAPTION_H
    scale = min(inner / max(h, w, 1), 1.0)
    if abs(scale - 1.0) > 0.02:
        img = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    canvas = np.full((cell, cell, 3), 30, dtype=np.uint8)
    h, w = img.shape[:2]
    x = (cell - w) // 2
    y = CAPTION_H + (inner - h) // 2
    y = max(CAPTION_H, min(y, cell - h))
    canvas[y : y + h, x : x + w] = img
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=20, help="crops per sheet")
    parser.add_argument("--cell", type=int, default=288, help="cell size in px")
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    items = _load_manifest(args.manifest)
    crops_root = args.manifest.parent
    args.out.mkdir(parents=True, exist_ok=True)

    cols = max(1, min(args.cols, args.cells))
    rows = (args.cells + cols - 1) // cols
    sheet_w = cols * args.cell
    sheet_h = rows * args.cell

    sheets = []
    sheet_of_id = {}
    for start in range(0, len(items), args.cells):
        chunk = items[start : start + args.cells]
        sheet_idx = start // args.cells + 1
        sheet_name = f"sheet_{sheet_idx:04d}.png"
        canvas = np.full((sheet_h, sheet_w, 3), 20, dtype=np.uint8)
        for i, item in enumerate(chunk):
            r, c = divmod(i, cols)
            cell = _cell_image(crops_root / item["crop"], args.cell)
            canvas[r * args.cell : (r + 1) * args.cell,
                   c * args.cell : (c + 1) * args.cell] = cell
            caption = f"{item['id']} {item['category'][0]}"
            cv2.putText(
                canvas, caption,
                (c * args.cell + 2, r * args.cell + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(args.out / sheet_name), canvas)
        sheets.append({"sheet": sheet_name, "ids": [i["id"] for i in chunk]})
        for item in chunk:
            sheet_of_id[item["id"]] = sheet_name

    (args.out / "sheets.jsonl").write_text(
        "\n".join(json.dumps(s, separators=(",", ":")) for s in sheets) + "\n",
        encoding="utf-8",
    )
    (args.out / "sheet_of_id.jsonl").write_text(
        "\n".join(json.dumps({k: v}, separators=(",", ":")) for k, v in sorted(sheet_of_id.items())) + "\n",
        encoding="utf-8",
    )
    print(f"{len(sheets)} sheets, {len(items)} cells -> {args.out}")


if __name__ == "__main__":
    main()
