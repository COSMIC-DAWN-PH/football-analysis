"""Prepare ball-review manifests: report, merge dual-model labels, order review.

The human reviews EVERY crop in eval/ball_crops/<src>/crops/. The dual models
(qwen3.7plus / gpt5.6luna) each write a plain label file - one JSON line per
item, {"id": "<item-id>", "label": "ball"|"not_ball"|"null", "reason": "..."}.
This tool merges those files back into candidates.jsonl and produces the
priority-ordered review lists (disagreements first).

Usage:
  python tools/prefill_ball_labels.py --report --manifest eval/ball_crops/demo4/candidates.jsonl
  python tools/prefill_ball_labels.py --merge-qwen eval/ball_crops/demo4/qwen_labels.jsonl \
      --merge-luna eval/ball_crops/demo4/luna_labels.jsonl \
      --manifest eval/ball_crops/demo4/candidates.jsonl
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


def _load_labels(path: Path) -> dict[str, tuple[str, str]]:
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
    qwen = _load_labels(args.merge_qwen) if args.merge_qwen else {}
    luna = _load_labels(args.merge_luna) if args.merge_luna else {}
    by_id = {i["id"]: i for i in items}
    missing_qwen = [iid for iid in by_id if iid not in qwen]
    missing_luna = [iid for iid in by_id if iid not in luna]
    if missing_qwen:
        print(f"WARNING: qwen labels missing for {len(missing_qwen)} items")
    if missing_luna:
        print(f"WARNING: luna labels missing for {len(missing_luna)} items")

    agree = disagree = only_qwen = only_luna = 0
    order = []
    for item in items:
        iid = item["id"]
        ql, qr = qwen.get(iid, ("null", ""))
        ll, lr = luna.get(iid, ("null", ""))
        item["qwen_label"] = ql
        item["qwen_reason"] = qr
        item["luna_label"] = ll
        item["luna_reason"] = lr
        cat_rank = REVIEW_ORDER.get(item["category"], 9)
        conf = item["conf"] if item["conf"] is not None else -1.0
        if iid in qwen and iid in luna:
            if ql == ll:
                agree += 1
                item["priority"] = cat_rank
                order.append((cat_rank, conf, iid))
            else:
                disagree += 1
                item["priority"] = -1
                order.append((-1, conf, iid))
        elif iid in qwen:
            only_qwen += 1
            item["priority"] = cat_rank
            order.append((cat_rank, conf, iid))
        elif iid in luna:
            only_luna += 1
            item["priority"] = cat_rank
            order.append((cat_rank, conf, iid))
        else:
            item["priority"] = cat_rank
            order.append((cat_rank, conf, iid))

    args.manifest.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False, separators=(",", ":")) for i in items) + "\n",
        encoding="utf-8",
    )
    print(f"merged into {args.manifest}: "
          f"agree={agree} disagree={disagree} qwen_only={only_qwen} luna_only={only_luna}")

    disagreements = [
        iid for i in items if i["qwen_label"] != i["luna_label"]
        and i["qwen_label"] is not None and i["luna_label"] is not None
    ]
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
    parser.add_argument("--merge-qwen", type=Path, default=None)
    parser.add_argument("--merge-luna", type=Path, default=None)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if args.report and not (args.merge_qwen or args.merge_luna):
        _report(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
