"""Validate and expand keyframe ball annotations into per-frame GT.

Reads eval/ball_gt/<src>/annotations.jsonl (keyframes with seg), linearly
interpolates bbox between consecutive keyframes of the SAME segment when the
gap <= MAX_GAP_FRAMES (default 2s), and writes per-frame GT to
eval/ball_gt/<src>/frames_gt.jsonl.

The source annotation file is never rewritten.  ``--repair-overlapping-segments``
can safely repair segment ids whose time ranges overlap (impossible for the
single-ball GT) and optionally write the normalized keyframes to a separate
file.

Usage:
  python tools/interp_ball_gt.py --src demo4 --max-gap 60
  python tools/interp_ball_gt.py --src demo4 \
      --repair-overlapping-segments \
      --normalized-annotations-out eval/ball_gt/demo4/annotations.optimized.jsonl
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FPS = 29.97


def _read_annotations(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        missing = {"frame", "bbox", "seg"} - row.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
            )
        frame = int(row["frame"])
        if frame < 0:
            raise ValueError(f"{path}:{line_number}: negative frame {frame}")
        if frame in seen_frames:
            raise ValueError(f"{path}:{line_number}: duplicate frame {frame}")
        seen_frames.add(frame)
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{path}:{line_number}: bbox must contain four values")
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: bbox contains a non-number") from error
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f"{path}:{line_number}: bbox contains a non-finite value")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{path}:{line_number}: bbox outside {width}x{height}: {bbox}"
            )
        rows.append(
            {
                "frame": frame,
                "t": round(frame / FPS, 4),
                "bbox": [x1, y1, x2, y2],
                "seg": int(row["seg"]),
            }
        )
    return sorted(rows, key=lambda row: row["frame"])


def _segment_intervals(rows: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    frames_by_segment: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        frames_by_segment[row["seg"]].append(row["frame"])
    return {
        segment: (min(frames), max(frames))
        for segment, frames in frames_by_segment.items()
    }


def _overlapping_segment_pairs(
    intervals: dict[int, tuple[int, int]],
) -> list[tuple[int, int]]:
    segments = sorted(intervals)
    overlaps: list[tuple[int, int]] = []
    for index, first in enumerate(segments):
        first_start, first_end = intervals[first]
        for second in segments[index + 1 :]:
            second_start, second_end = intervals[second]
            if max(first_start, second_start) <= min(first_end, second_end):
                overlaps.append((first, second))
    return overlaps


def _repair_overlapping_segments(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Merge temporally overlapping ids, then renumber groups chronologically."""
    intervals = _segment_intervals(rows)
    parent = {segment: segment for segment in intervals}

    def find(segment: int) -> int:
        while parent[segment] != segment:
            parent[segment] = parent[parent[segment]]
            segment = parent[segment]
        return segment

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in _overlapping_segment_pairs(intervals):
        union(first, second)

    groups: dict[int, list[int]] = defaultdict(list)
    for segment in intervals:
        groups[find(segment)].append(segment)
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: min(intervals[segment][0] for segment in group),
    )
    mapping = {
        old_segment: new_segment
        for new_segment, group in enumerate(ordered_groups)
        for old_segment in group
    }
    normalized = [dict(row, seg=mapping[row["seg"]]) for row in rows]
    return normalized, mapping


def _interpolate(
    rows: list[dict[str, Any]], max_gap_frames: int
) -> list[dict[str, Any]]:
    out = {
        row["frame"]: {
            "frame": row["frame"],
            "t": row["t"],
            "bbox": row["bbox"],
            "seg": row["seg"],
            "interp": False,
        }
        for row in rows
    }
    for first, second in zip(rows, rows[1:]):
        first_frame, second_frame = first["frame"], second["frame"]
        if first["seg"] != second["seg"]:
            continue
        if second_frame - first_frame > max_gap_frames:
            continue
        for frame in range(first_frame + 1, second_frame):
            ratio = (frame - first_frame) / (second_frame - first_frame)
            out[frame] = {
                "frame": frame,
                "t": round(frame / FPS, 4),
                "bbox": [
                    first["bbox"][index]
                    + (second["bbox"][index] - first["bbox"][index]) * ratio
                    for index in range(4)
                ],
                "seg": first["seg"],
                "interp": True,
            }
    return [out[frame] for frame in sorted(out)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--max-gap", type=int, default=60, help="max frames to interpolate across (default 60 = 2s)")
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--repair-overlapping-segments",
        action="store_true",
        help="merge segment ids whose temporal ranges overlap and renumber from zero",
    )
    parser.add_argument(
        "--normalized-annotations-out",
        type=Path,
        default=None,
        help="write validated/normalized keyframes to a separate file",
    )
    args = parser.parse_args()

    gt_dir = ROOT / "eval" / "ball_gt" / args.src
    ann_path = args.annotations or (gt_dir / "annotations.jsonl")
    if not ann_path.is_absolute():
        ann_path = ROOT / ann_path
    if not ann_path.is_file():
        print(f"no annotations: {ann_path}")
        sys.exit(1)

    rows = _read_annotations(ann_path, args.width, args.height)
    intervals = _segment_intervals(rows)
    overlaps = _overlapping_segment_pairs(intervals)
    if overlaps:
        print(f"warning: overlapping segment ranges: {overlaps}")
    if args.repair_overlapping_segments:
        rows, mapping = _repair_overlapping_segments(rows)
        print(f"segment mapping: {mapping}")

    suspicious_boxes = []
    for row in rows:
        width = row["bbox"][2] - row["bbox"][0]
        height = row["bbox"][3] - row["bbox"][1]
        ratio = width / height
        if not 0.75 <= ratio <= 4 / 3:
            suspicious_boxes.append(row["frame"])
    if suspicious_boxes:
        print(
            "warning: strongly non-square boxes need visual review at frames: "
            + ",".join(map(str, suspicious_boxes))
        )

    if args.normalized_annotations_out:
        normalized_path = args.normalized_annotations_out
        if not normalized_path.is_absolute():
            normalized_path = ROOT / normalized_path
        _write_jsonl(normalized_path, rows)
        print(f"normalized keyframes -> {normalized_path}")

    expanded = _interpolate(rows, args.max_gap)
    out_path = args.out or (gt_dir / "frames_gt.jsonl")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    _write_jsonl(out_path, expanded)
    n_interp = sum(1 for row in expanded if row["interp"])
    print(
        f"{args.src}: {len(expanded)} frames GT "
        f"({len(rows)} keyframes, {n_interp} interpolated) -> {out_path}"
    )


if __name__ == "__main__":
    main()
