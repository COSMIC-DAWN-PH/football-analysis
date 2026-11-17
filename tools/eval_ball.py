"""Evaluate ball detection/tracking against reviewed crops or frame-level GT.

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
  python tools/eval_ball.py --gt eval/ball_gt/demo4/frames_gt.jsonl \
      --tracks output_videos/demo4/ball/ball_tracks.jsonl \
      --out eval/ball_gt/demo4/baseline.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from math import hypot
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fraction(tp: int, total: int) -> float:
    return round(tp / total, 4) if total else None


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(mean(values), 4) if values else None,
        "median": round(median(values), 4) if values else None,
    }


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2,
        (float(bbox[1]) + float(bbox[3])) / 2,
    )


def _center_distance(first: list[float], second: list[float]) -> float:
    first_x, first_y = _bbox_center(first)
    second_x, second_y = _bbox_center(second)
    return hypot(first_x - second_x, first_y - second_y)


def _center_group_report(
    items: list[dict[str, Any]], details_limit: int
) -> dict[str, Any]:
    track_matches = [item for item in items if item["track_center_match"]]
    observed_matches = [
        item
        for item in track_matches
        if item["track"] is not None and item["track"].get("observed")
    ]
    confirmed_matches = [
        item
        for item in track_matches
        if item["track"] is not None and item["track"].get("track_confirmed")
    ]
    candidate_matches = [item for item in items if item["candidate_center_match"]]
    no_candidate = [item for item in items if not item["candidate_center_match"]]
    tracker_selection_misses = [
        item
        for item in items
        if item["candidate_center_match"] and not item["track_center_match"]
    ]
    wrong_track = [
        item
        for item in items
        if item["track"] is not None and not item["track_center_match"]
    ]
    absent_track = [item for item in items if item["track"] is None]
    matched_segments = [
        int(item["track"]["track_segment"])
        for item in track_matches
        if item["track"].get("track_segment") is not None
    ]
    return {
        "rule": "predicted center within the GT box (half-diagonal radius)",
        "gt_frames": len(items),
        "track_matched": len(track_matches),
        "track_recall": _fraction(len(track_matches), len(items)),
        "observed_track_matched": len(observed_matches),
        "observed_track_recall": _fraction(len(observed_matches), len(items)),
        "confirmed_track_matched": len(confirmed_matches),
        "confirmed_track_recall": _fraction(len(confirmed_matches), len(items)),
        "candidate_matched": len(candidate_matches),
        "candidate_recall": _fraction(len(candidate_matches), len(items)),
        "missed_no_matching_candidate": len(no_candidate),
        "missed_tracker_selection": len(tracker_selection_misses),
        "wrong_track_on_gt_frame": len(wrong_track),
        "no_track_on_gt_frame": len(absent_track),
        "matched_track_center_distance_px": _summary(
            [item["track_center_distance"] for item in track_matches]
        ),
        "matched_candidate_center_distance_px": _summary(
            [item["candidate_center_distance"] for item in candidate_matches]
        ),
        "matched_track_segments": len(set(matched_segments)),
        "sample_frames": {
            "no_matching_candidate": [
                item["frame"] for item in no_candidate[:details_limit]
            ],
            "tracker_selection_miss": [
                item["frame"] for item in tracker_selection_misses[:details_limit]
            ],
            "wrong_track": [item["frame"] for item in wrong_track[:details_limit]],
        },
    }


def _frame_group_report(
    items: list[dict[str, Any]], iou_threshold: float, details_limit: int
) -> dict[str, Any]:
    track_matches = [item for item in items if item["track_iou"] >= iou_threshold]
    observed_matches = [
        item
        for item in track_matches
        if item["track"] is not None and item["track"].get("observed")
    ]
    confirmed_matches = [
        item
        for item in track_matches
        if item["track"] is not None and item["track"].get("track_confirmed")
    ]
    candidate_matches = [
        item for item in items if item["candidate_iou"] >= iou_threshold
    ]
    no_candidate = [
        item for item in items if item["candidate_iou"] < iou_threshold
    ]
    tracker_selection_misses = [
        item
        for item in items
        if item["candidate_iou"] >= iou_threshold
        and item["track_iou"] < iou_threshold
    ]
    wrong_track = [
        item
        for item in items
        if item["track"] is not None and item["track_iou"] < iou_threshold
    ]
    absent_track = [item for item in items if item["track"] is None]
    matched_segments = [
        int(item["track"]["track_segment"])
        for item in track_matches
        if item["track"].get("track_segment") is not None
    ]
    return {
        "gt_frames": len(items),
        "track_matched": len(track_matches),
        "track_recall": _fraction(len(track_matches), len(items)),
        "observed_track_matched": len(observed_matches),
        "observed_track_recall": _fraction(len(observed_matches), len(items)),
        "confirmed_track_matched": len(confirmed_matches),
        "confirmed_track_recall": _fraction(len(confirmed_matches), len(items)),
        "candidate_matched": len(candidate_matches),
        "candidate_recall": _fraction(len(candidate_matches), len(items)),
        "missed_no_matching_candidate": len(no_candidate),
        "missed_tracker_selection": len(tracker_selection_misses),
        "wrong_track_on_gt_frame": len(wrong_track),
        "no_track_on_gt_frame": len(absent_track),
        "matched_track_iou": _summary(
            [item["track_iou"] for item in track_matches]
        ),
        "matched_candidate_iou": _summary(
            [item["candidate_iou"] for item in candidate_matches]
        ),
        "matched_track_segments": len(set(matched_segments)),
        "sample_frames": {
            "no_matching_candidate": [
                item["frame"] for item in no_candidate[:details_limit]
            ],
            "tracker_selection_miss": [
                item["frame"] for item in tracker_selection_misses[:details_limit]
            ],
            "wrong_track": [item["frame"] for item in wrong_track[:details_limit]],
        },
    }


def _frame_gt_report(
    gt: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    iou_threshold: float,
    details_limit: int,
) -> dict[str, Any]:
    tracks_by_frame = {int(row["frame"]): row for row in tracks}
    evaluated: list[dict[str, Any]] = []
    missing_track_rows = 0
    for gt_row in gt:
        frame = int(gt_row["frame"])
        track_row = tracks_by_frame.get(frame)
        if track_row is None:
            missing_track_rows += 1
            track = None
            candidates: list[list[float]] = []
        else:
            track = track_row.get("track")
            candidates = track_row.get("candidates") or []
        track_iou = (
            _bbox_iou(gt_row["bbox"], track["bbox"])
            if track is not None and track.get("bbox")
            else 0.0
        )
        candidate_iou = max(
            (_bbox_iou(gt_row["bbox"], candidate[:4]) for candidate in candidates),
            default=0.0,
        )
        gt_width = float(gt_row["bbox"][2]) - float(gt_row["bbox"][0])
        gt_height = float(gt_row["bbox"][3]) - float(gt_row["bbox"][1])
        center_radius = max(4.0, hypot(gt_width, gt_height) / 2)
        track_center_distance = (
            _center_distance(gt_row["bbox"], track["bbox"])
            if track is not None and track.get("bbox")
            else None
        )
        candidate_center_distance = min(
            (
                _center_distance(gt_row["bbox"], candidate[:4])
                for candidate in candidates
            ),
            default=None,
        )
        evaluated.append(
            {
                "frame": frame,
                "interp": bool(gt_row.get("interp", False)),
                "gt_segment": int(gt_row.get("seg", 0)),
                "track": track,
                "track_iou": track_iou,
                "candidate_iou": candidate_iou,
                "track_center_distance": track_center_distance,
                "candidate_center_distance": candidate_center_distance,
                "track_center_match": (
                    track_center_distance is not None
                    and track_center_distance <= center_radius
                ),
                "candidate_center_match": (
                    candidate_center_distance is not None
                    and candidate_center_distance <= center_radius
                ),
            }
        )

    keyframes = [item for item in evaluated if not item["interp"]]
    interpolated = [item for item in evaluated if item["interp"]]
    segment_ids = sorted({item["gt_segment"] for item in evaluated})
    return {
        "iou_threshold": iou_threshold,
        "scope_note": (
            "Recall/correctness only on frames with visible-ball GT. Precision on "
            "unlabelled frames is intentionally not reported for partial annotation."
        ),
        "missing_track_file_rows": missing_track_rows,
        "all": {
            "iou": _frame_group_report(evaluated, iou_threshold, details_limit),
            "center": _center_group_report(evaluated, details_limit),
        },
        "keyframes": {
            "iou": _frame_group_report(keyframes, iou_threshold, details_limit),
            "center": _center_group_report(keyframes, details_limit),
        },
        "interpolated": {
            "iou": _frame_group_report(
                interpolated, iou_threshold, details_limit
            ),
            "center": _center_group_report(interpolated, details_limit),
        },
        "segments": {
            str(segment): {
                "all": {
                    "iou": _frame_group_report(
                        [
                            item
                            for item in evaluated
                            if item["gt_segment"] == segment
                        ],
                        iou_threshold,
                        details_limit,
                    ),
                    "center": _center_group_report(
                        [
                            item
                            for item in evaluated
                            if item["gt_segment"] == segment
                        ],
                        details_limit,
                    ),
                },
                "keyframes": {
                    "iou": _frame_group_report(
                        [
                            item
                            for item in keyframes
                            if item["gt_segment"] == segment
                        ],
                        iou_threshold,
                        details_limit,
                    ),
                    "center": _center_group_report(
                        [
                            item
                            for item in keyframes
                            if item["gt_segment"] == segment
                        ],
                        details_limit,
                    ),
                },
            }
            for segment in segment_ids
        },
    }


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
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--details-limit", type=int, default=50)
    args = parser.parse_args()
    if args.manifest is None and args.gt is None:
        parser.error("at least one of --manifest or --gt is required")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if args.gt is not None and not args.gt.is_file():
        raise FileNotFoundError(f"Missing frame GT: {args.gt}")
    if not args.tracks.is_file():
        raise FileNotFoundError(f"Missing tracks: {args.tracks}")

    tracks = _load(args.tracks)
    report: dict[str, Any] = {}
    if args.manifest is not None:
        report.update(_report(_load(args.manifest), tracks))
    if args.gt is not None:
        report["frame_gt"] = _frame_gt_report(
            _load(args.gt), tracks, args.iou, args.details_limit
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
