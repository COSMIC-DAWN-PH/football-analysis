"""Build a curated labeling subset from raw1/raw2 referee candidate manifests.

Selects the most informative crops: all boundary and dark entries (ambiguous
hue / dark jerseys), a sample of yellow (referee candidates) and ref_cls
entries, skipping the maroon/navy controls (demo2 already covers regression).

Usage:
  python tools/make_raw_labeling_subset.py --out eval/referee_crops/raw_labeling_subset.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAW_DIR = Path("eval/referee_crops")
QUOTA = {"boundary": 1000, "dark": 1000, "yellow": 20, "ref_cls": 12}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_DIR / "raw_labeling_subset.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_category = {}
    for source in ("raw1", "raw2"):
        manifest = RAW_DIR / source / "candidates.jsonl"
        if not manifest.is_file():
            print(f"missing {manifest}")
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            meta = json.loads(line)
            meta["_manifest"] = f"eval/referee_crops/{source}/candidates.jsonl"
            by_category.setdefault(meta["category"], []).append(meta)

    picked = []
    for category, quota in QUOTA.items():
        entries = by_category.get(category, [])
        rng.shuffle(entries)
        picked.extend(entries[:quota])

    # Order for convenient labeling: boundary/dark first (null labels), then
    # yellow, then ref_cls.
    order = {"boundary": 0, "dark": 1, "yellow": 2, "ref_cls": 3}
    picked.sort(key=lambda m: order.get(m["category"], 9))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in picked) + "\n",
        encoding="utf-8",
    )
    counts = {}
    for m in picked:
        counts[m["category"]] = counts.get(m["category"], 0) + 1
    print(f"subset: {len(picked)} crops {counts} -> {args.out}")


if __name__ == "__main__":
    main()
