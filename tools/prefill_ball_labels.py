"""Prepare ball-review manifests: report, merge dual-model labels, order review.

The human reviews EVERY crop in eval/ball_crops/<src>/crops/. The dual models
(qwen3.7plus / gpt5.6luna) each write a plain label file - one JSON line per
item, {"id": "<item-id>", "label": "ball"|"not_ball"|"null", "reason": "..."}.
This tool merges those files back into candidates.jsonl and produces the
priority-ordered review lists (disagreements first).

Usage:
  python tools/prefill_ball_labels.py --report --manifest eval/ball_crops/demo4/candidates.jsonl
  python tools/prefill_ball_labels.py --manifest eval/ball_crops/demo4/candidates.jsonl \
      --merge-labels eval/ball_crops/demo4/luna_labels.jsonl luna_label \
      --merge-labels eval/ball_crops/demo4/kimi_labels.jsonl kimi_label
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VALID_LABELS = {"ball", "not_ball", "null"}
REVIEW_ORDER = {
    "unconfirmed": 0,
    "bridge_sweep": 1,
    "gap_sweep": 2,
    "global_sweep": 3,
    "confirmed": 4,
}


def _load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_labels(path) -> dict[str, tuple[str, str]]:
    path = Path(path)
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        label = d.get("label")
        if label not in VALID_LABELS:
            raise ValueError(f"{path}: invalid label {label!r} for {d.get('id')}")
        labels[d["id"]] = (label, d.get("reason", ""))
    return labels


def _report(args) -> None:
    items = _load_manifest(args.manifest)
    by_cat = Counter(i["category"] for i in items)
    confs = [i["conf"] for i in items if i["conf"] is not None]
    confs.sort()
    print(f"items: {len(items)}")
    for cat in sorted(by_cat):
        print(f"  {cat}: {by_cat[cat]}")
    if confs:
        print(f"conf range: {confs[0]:.3f}..{confs[-1]:.3f}, "
              f"median {confs[len(confs) // 2]:.3f}")
    missing_crops = [i["id"] for i in items
                     if not (args.manifest.parent / i["crop"]).is_file()]
    if missing_crops:
        print(f"MISSING CROPS ({len(missing_crops)}): {missing_crops[:5]}...")
    else:
        print("all crops present")


def _merge(args) -> None:
    items = _load_manifest(args.manifest)
    sources = {}
    for path, field in args.merge_labels or []:
        sources[field] = (_load_labels(path), path)
    by_id = {i["id"]: i for i in items}
    for field, (labels, path) in sources.items():
        missing = [iid for iid in by_id if iid not in labels]
        if missing:
            print(f"WARNING: {field} labels missing for {len(missing)} items ({path})")

    agree = disagree = single = 0
    order = []
    fields = list(sources.keys())
    for item in items:
        iid = item["id"]
        per = {}
        for field in fields:
            label, reason = sources[field][0].get(iid, (None, ""))
            item[field] = label
            if reason:
                item[f"{field}_reason"] = reason
            per[field] = label
        cat_rank = REVIEW_ORDER.get(item["category"], 9)
        conf = item["conf"] if item["conf"] is not None else -1.0
        present = [f for f in fields if per.get(f) is not None]
        if len(present) >= 2:
            first, *rest = [per[f] for f in present]
            if all(v == first for v in rest):
                agree += 1
                item["priority"] = cat_rank
                order.append((cat_rank, conf, iid))
            else:
                disagree += 1
                item["priority"] = -1
                order.append((-1, conf, iid))
        else:
            single += 1
            item["priority"] = cat_rank
            order.append((cat_rank, conf, iid))

    args.manifest.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False, separators=(",", ":")) for i in items) + "\n",
        encoding="utf-8",
    )
    print(f"merged into {args.manifest}: "
          f"agree={agree} disagree={disagree} single={single} "
          f"(fields: {', '.join(fields) or 'none'})")

    disagreements = []
    for item in items:
        vals = {item[f] for f in fields if item.get(f) is not None}
        if len(vals) > 1:
            disagreements.append(item["id"])
    d_path = args.manifest.parent / "disagreements.jsonl"
    d_path.write_text("\n".join(disagreements) + "\n", encoding="utf-8")
    print(f"disagreement ids -> {d_path} ({len(disagreements)} items, review first)")

    ordered = sorted(order, key=lambda t: (t[0], t[1]))
    o_path = args.manifest.parent / "review_order.txt"
    o_path.write_text("\n".join(iid for _, _, iid in ordered) + "\n", encoding="utf-8")
    print(f"review order -> {o_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--merge-labels", nargs=2, action="append", metavar=("PATH", "FIELD"),
        help="Merge a model label file into FIELD (repeatable), "
             "e.g. --merge-labels luna_labels.jsonl luna_label "
             "--merge-labels kimi_labels.jsonl kimi_label",
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if args.report and not args.merge_labels:
        _report(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
