"""Copy the curated subset crops into one folder for convenient labeling.

Usage:
  python tools/package_raw_labeling_subset.py --subset eval/referee_crops/raw_labeling_subset.jsonl \
      --out-dir eval/referee_crops/raw_labeling
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=Path("eval/referee_crops/raw_labeling_subset.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/referee_crops/raw_labeling"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for line in args.subset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        manifest = Path(meta.pop("_manifest"))
        src = manifest.parent / meta["file"]
        if not src.is_file():
            print(f"missing {src}")
            continue
        target = args.out_dir / meta["file"]
        target.write_bytes(src.read_bytes())
        entries.append(meta)

    (args.out_dir / "candidates.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in entries) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "README.md").write_text(
        "# raw1/raw2 精选标注集（116 张）\n\n"
        "只标这一个目录即可。candidates.jsonl 已预填 auto 猜测，看图纠错：\n"
        "- `referee`：黄衣裁判\n"
        "- `maroon` / `navy`：栗色/藏青球员\n"
        "- `maroon_gk` / `navy_gk`：对应球队门将（navy 门将深紫、maroon 门将黑色）\n"
        "- 看不清/歧义：改回 `null`\n\n"
        "重点：boundary/dark 两类（共 84 张）大多未预填（null），是最需要你判断的；"
        "yellow/ref_cls 是抽查确认。\n",
        encoding="utf-8",
    )
    print(f"packaged {len(entries)} crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
