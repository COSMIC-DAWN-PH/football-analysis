"""Evaluate ball detection/tracking against the human-reviewed label set.

Reads a review manifest (candidates.jsonl with manual_label filled in) plus
the source ball_tracks.jsonl and reports:

  Crop level   candidate precision by category (null labels excluded),
               and a sampled detection-recall proxy from the fn_sweep crops
               (bridge/gap/global items labelled ball are detections the
               detector missed)
  Track level  segment statistics: FP segments (all reviewed crops not_ball),
               segment precision, mean/median segment length, confirmation
               latency (frames from segment start to first confirmed frame),
               gap counts (segment breaks where the ball was actually there)

Usage:
  python tools/eval_ball.py --manifest eval/ball_crops/demo4/candidates.jsonl \
      --tracks output_videos/demo4/ball/ball_tracks.jsonl --out eval/ball_crops/demo4/baseline.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fraction(tp: int, total: int) -> float:
    return round(tp / total, 4) if total else None


def _report(manifest: list[dict], tracks: list[dict]) -> dict:
    total = len(manifest)
    labeled = [i for i in manifest if i.get("manual_label") is not None]
    nulls = [i for i in manifest if i.get("manual_label") is None]
    balls = [i for i in labeled if i["manual_label"] == "ball"]
    not_balls = [i for i in labeled if i["manual_label"] == "not_ball"]

    by_cat_labeled = Counter(i["category"] for i in labeled)
    by_cat_ball = Counter(i["category"] for i in balls)
    by_cat_null = Counter(i["category"] for i in nulls)

    candidate_cats = ("confirmed", "unconfirmed")
    cand_total = sum(by_cat_labeled[c] for c in candidate_cats)
    cand_ball = sum(by_cat_ball[c] for c in candidate_cats)
    sweep_cats = ("bridge_sweep", "gap_sweep", "global_sweep")
    sweep_ball = sum(by_cat_ball[c] for c in sweep_cats)
    sweep_total = sum(by_cat_labeled[c] for c in sweep_cats)

    crop_metrics = {
        "review_progress": {
            "total": total,
            "labeled": len(labeled),
            "null": len(nulls),
            "labeled_fraction": round(len(labeled) / total, 4) if total else None,
        },
        "candidate_precision": _fraction(cand_ball, cand_total),
        "candidate_precision_by_category": {
            c: {
                "labeled": by_cat_labeled[c],
                "ball": by_cat_ball[c],
                "precision": _fraction(by_cat_ball[c], by_cat_labeled[c]),
            }
            for c in candidate_cats
        },
        "sweep_false_negatives": {
            c: {"labeled": by_cat_labeled[c], "ball_missed": by_cat_ball[c]}
            for c in sweep_cats
        },
        "sampled_recall_proxy": _fraction(
            cand_ball, cand_ball + sweep_ball
        ) if cand_ball + sweep_ball else None,
        "null_by_category": dict(by_cat_null),
    }

    # --- track level -----------------------------------------------------------
    seg_frames = defaultdict(list)       # segment -> [frame numbers]
    seg_confirmed_frames = defaultdict(list)
    for d in tracks:
        t = d.get("track")
        if t and t.get("track_segment"):
            seg = t["track_segment"]
            seg_frames[seg].append(d["frame"])
            if t.get("track_confirmed"):
                seg_confirmed_frames[seg].append(d["frame"])
    seg_ids = sorted(seg_frames)
    confirmed_by_seg = defaultdict(list)
    for i in manifest:
        if i.get("segment") is not None and i["category"] == "confirmed":
            confirmed_by_seg[i["segment"]].append(i)

    fp_segments = []
    tp_segments = []
    for seg in seg_ids:
        items = confirmed_by_seg.get(seg, [])
        labeled_items = [i for i in items if i.get("manual_label") is not None]
        if not labeled_items:
            continue
        if any(i["manual_label"] == "ball" for i in labeled_items):
            tp_segments.append(seg)
        else:
            fp_segments.append(seg)

    lengths = [len(seg_frames[s]) for s in seg_ids]
    latencies = []
    for s in seg_ids:
        first = min(seg_frames[s])
        conf = seg_confirmed_frames.get(s)
        if conf:
            latencies.append(min(conf) - first)

    labeled_segs = len(fp_segments) + len(tp_segments)
    track_metrics = {
        "segments_total": len(seg_ids),
        "segments_labeled": labeled_segs,
        "true_segments": len(tp_segments),
        "false_segments": len(fp_segments),
        "segment_precision": _fraction(len(tp_segments), labeled_segs),
        "segment_length": {
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else None,
            "median": sorted(lengths)[len(lengths) // 2] if lengths else None,
        },
        "confirmation_latency_frames": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "median": sorted(latencies)[len(latencies) // 2] if latencies else None,
            "n": len(latencies),
        },
    }

    gap_balls = [i for i in balls if i["category"] == "gap_sweep"]
    track_metrics["gap_breaks_with_ball"] = len(gap_balls)

    return {"crop": crop_metrics, "track": track_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if not args.tracks.is_file():
        raise FileNotFoundError(f"Missing tracks: {args.tracks}")

    report = _report(_load(args.manifest), _load(args.tracks))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
