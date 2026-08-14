"""Extract referee-labelling candidate crops from object tracks + video.

Candidates are chosen per track (aggregated across frames) so a person appears
at most --max-per-track times, in their clearest frames. Categories:

  yellow    player-class tracks whose median jersey hue is 15-45 (referee)
  dark      player-class tracks with valid jersey stats in <30% of frames
  ref_cls   referee-class tracks (true referees AND misclassified players)
  boundary  player tracks with median hue 45-95 (between yellow and navy)
  controls  stratified sample of clear maroon/navy players

Usage:
  python tools/extract_referee_candidates.py \
      --video output_videos/demo2-30s-test/demo2-30s-test-input.mp4 \
      --tracks output_videos/demo2-30s-test/raw/object_tracks.jsonl \
      --source demo2 --out eval/referee_crops/demo2

Verdict mode (verify the new assigner's decisions against a replayed tracks
file): add --verdict-tracks <replayed jsonl> to extract crops of every track
whose class was changed by the referee logic (player->referee flags and
referee->club restorations).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from club_assignment.club_assigner import ClubAssigner

YELLOW_LO, YELLOW_HI = 15.0, 45.0
BOUNDARY_LO, BOUNDARY_HI = 45.0, 95.0
NAVY_LO, NAVY_HI = 95.0, 135.0
MAROON_LO = 135.0  # wraps through 180/0
MIN_VALID_RATIO = 0.30
MIN_BBOX_W, MIN_BBOX_H = 20.0, 55.0
MARGIN = 0.15
MIN_FRAME_GAP = 15
UPSCALE_TO = 96


def _rgb(s: str):
    r, g, b = (int(v) for v in s.split(","))
    return (r, g, b)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--source", required=True, help="e.g. demo2 / raw1 / raw2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--club1", type=_rgb, default=(120, 37, 66))
    parser.add_argument("--club2", type=_rgb, default=(31, 72, 127))
    parser.add_argument("--max-per-track", type=int, default=3)
    parser.add_argument("--max-referee-candidates", type=int, default=80)
    parser.add_argument("--max-controls", type=int, default=20)
    parser.add_argument("--max-boundary", type=int, default=15)
    parser.add_argument("--min-bbox-w", type=float, default=MIN_BBOX_W)
    parser.add_argument("--min-bbox-h", type=float, default=MIN_BBOX_H)
    parser.add_argument(
        "--verdict-tracks",
        type=Path,
        default=None,
        help="Replayed tracks JSONL (with referee flags); when given, only "
        "extract crops of tracks whose class was changed by the referee logic",
    )
    return parser


def _hue_zone(hue):
    if YELLOW_LO <= hue < YELLOW_HI:
        return "yellow"
    if BOUNDARY_LO <= hue < BOUNDARY_HI:
        return "boundary"
    if NAVY_LO <= hue < NAVY_HI:
        return "navy"
    return "maroon"


def main() -> None:
    args = _parser().parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Missing video: {args.video}")
    if not args.tracks.is_file():
        raise FileNotFoundError(f"Missing tracks: {args.tracks}")
    if args.verdict_tracks is not None:
        if not args.verdict_tracks.is_file():
            raise FileNotFoundError(f"Missing verdict tracks: {args.verdict_tracks}")
        _run_verdict(args)
        return

    assigner = ClubAssigner.__new__(ClubAssigner)
    lines = args.tracks.read_text(encoding="utf-8").splitlines()

    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.source}: {len(lines)} track lines, {total_frames} video frames")

    # Per (track_type, track_id): list of dicts {frame, bbox, hue, area, pixels}
    track_frames = defaultdict(list)
    for fi, line in enumerate(lines):
        d = json.loads(line)
        any_person = any(d.get(t) for t in ("player", "referee", "goalkeeper"))
        if not any_person:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        for track_type in ("player", "referee", "goalkeeper"):
            for track_id, item in d.get(track_type, {}).items():
                bbox = item["bbox"]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < args.min_bbox_w or h < args.min_bbox_h:
                    continue
                stats = assigner.extract_jersey_stats(frame, bbox)
                hue = None
                pixels = 0
                if stats is not None:
                    hue, _, _, pixels = stats
                track_frames[(track_type, int(track_id))].append(
                    {
                        "frame": fi,
                        "bbox": bbox,
                        "hue": hue if (hue is not None and pixels >= 30) else None,
                        "area": float(w * h),
                    }
                )
    cap.release()

    def median_hue(entries):
        hues = [e["hue"] for e in entries if e["hue"] is not None]
        return float(np.median(hues)) if hues else None

    def valid_ratio(entries):
        return sum(1 for e in entries if e["hue"] is not None) / max(1, len(entries))

    # Category buckets: list of (track_type, track_id, entries, reason)
    buckets = {
        "yellow": [],
        "dark": [],
        "ref_cls": [],
        "boundary": [],
        "navy_control": [],
        "maroon_control": [],
    }
    for (track_type, track_id), entries in track_frames.items():
        if track_type == "goalkeeper":
            continue
        mh = median_hue(entries)
        if track_type == "referee":
            buckets["ref_cls"].append((track_type, track_id, entries, f"ref_cls_h{mh}"))
        elif mh is None or valid_ratio(entries) < MIN_VALID_RATIO:
            buckets["dark"].append((track_type, track_id, entries, "dark"))
        else:
            zone = _hue_zone(mh)
            if zone == "yellow":
                buckets["yellow"].append((track_type, track_id, entries, f"hue{mh:.0f}"))
            elif zone == "boundary":
                buckets["boundary"].append((track_type, track_id, entries, f"hue{mh:.0f}"))
            elif zone == "navy":
                buckets["navy_control"].append((track_type, track_id, entries, f"hue{mh:.0f}"))
            else:
                buckets["maroon_control"].append((track_type, track_id, entries, f"hue{mh:.0f}"))

    # Priorities: referee candidates first (yellow > dark > ref_cls), then
    # boundary, then controls. Fill quotas.
    quota = {
        "yellow": args.max_referee_candidates,
        "dark": args.max_referee_candidates // 4,
        "ref_cls": args.max_referee_candidates,
        "boundary": args.max_boundary,
        "navy_control": args.max_controls // 2,
        "maroon_control": args.max_controls // 2,
    }
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    manifest = []
    used_frames = set()
    for category in (
        "yellow", "dark", "ref_cls", "boundary", "navy_control", "maroon_control",
    ):
        count = 0
        for track_type, track_id, entries, reason in buckets[category]:
            if count >= quota[category]:
                break
            picked = []
            for entry in sorted(entries, key=lambda e: -e["area"]):
                if len(picked) >= args.max_per_track:
                    break
                if not picked or all(abs(entry["frame"] - p["frame"]) >= MIN_FRAME_GAP for p in picked):
                    picked.append(entry)
            saved_this_track = 0
            for entry in picked:
                if saved_this_track >= args.max_per_track:
                    break
                fi = entry["frame"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = (int(v) for v in entry["bbox"])
                H, W = frame.shape[:2]
                mx = int((x2 - x1) * MARGIN)
                my = int((y2 - y1) * MARGIN)
                x1c = max(0, x1 - mx)
                y1c = max(0, y1 - my)
                x2c = min(W, x2 + mx)
                y2c = min(H, y2 + my)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue
                if crop.shape[1] < UPSCALE_TO:
                    scale = UPSCALE_TO / crop.shape[1]
                    crop = cv2.resize(
                        crop, (UPSCALE_TO, int(crop.shape[0] * scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                hue_str = f"{entry['hue']:.0f}" if entry["hue"] is not None else "na"
                tpref = "p" if track_type == "player" else "r"
                name = f"{args.source}_{tpref}{track_id}_f{fi}_h{hue_str}_{category}.png"
                cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                manifest.append(
                    {
                        "file": name,
                        "source": args.source,
                        "track_type": track_type,
                        "track_id": track_id,
                        "frame": fi,
                        "yolo_class": track_type,
                        "median_hue": entry["hue"],
                        "category": category,
                        "reason": reason,
                        "manual_label": None,
                    }
                )
                used_frames.add((category, track_type, track_id, fi))
                saved_this_track += 1
            count += 1
    cap.release()

    (out_dir / "candidates.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# 裁判候选标注说明\n\n"
        "请逐张查看本目录下的 PNG 图片，在 `candidates.jsonl` 中为每行填写 "
        "`manual_label`：\n\n"
        "- `referee`：裁判（黄衣或黑衣）\n"
        "- `maroon`：栗色/红队球员\n"
        "- `navy`：藏青/蓝队球员\n"
        "- 看不清/有歧义：保持 `null`（该样本不进评估）\n\n"
        "文件名格式：`{source}_{p|r}{track_id}_f{帧号}_h{躯干中位hue}_{category}.png`；"
        "p=player 类，r=referee 类。\n",
        encoding="utf-8",
    )

    counts = defaultdict(int)
    for m in manifest:
        counts[m["category"]] += 1
    print(f"{args.source}: extracted {len(manifest)} crops: {dict(counts)}")
    print(f"-> {out_dir}")


def _run_verdict(args) -> None:
    """Extract crops of every track whose class the referee logic changed."""
    assigner = ClubAssigner.__new__(ClubAssigner)
    lines = args.verdict_tracks.read_text(encoding="utf-8").splitlines()

    cap = cv2.VideoCapture(str(args.video))
    # Changed frames per (track_type, track_id): list of {frame, bbox, hue, area}
    changed = defaultdict(list)
    seen = defaultdict(set)
    for fi, line in enumerate(lines):
        d = json.loads(line)
        relevant = False
        for track_type in ("player", "referee", "goalkeeper"):
            if d.get(track_type):
                relevant = True
                break
        if not relevant:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        for track_type in ("player", "referee", "goalkeeper"):
            for track_id, item in d.get(track_type, {}).items():
                key = (track_type, int(track_id))
                is_changed = (
                    (track_type != "referee" and item.get("referee"))
                    or (track_type == "referee" and item.get("club") is not None)
                )
                if not is_changed:
                    continue
                bbox = item["bbox"]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < args.min_bbox_w or h < args.min_bbox_h:
                    continue
                stats = assigner.extract_jersey_stats(frame, bbox)
                hue = None
                if stats is not None and stats[3] >= 30:
                    hue = float(stats[0])
                changed[key].append(
                    {
                        "frame": fi,
                        "bbox": bbox,
                        "hue": hue,
                        "area": float(w * h),
                    }
                )
                seen[key].add(fi)
    cap.release()

    out_dir = args.out / "verdict"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    manifest = []
    count = 0
    for key in sorted(changed, key=lambda k: -len(changed[k])):
        track_type, track_id = key
        entries = changed[key]
        if len(entries) < 3:
            continue
        if count >= args.max_referee_candidates:
            break
        category = "flag_player" if track_type != "referee" else "restore_referee"
        picked = []
        for entry in sorted(entries, key=lambda e: -e["area"]):
            if len(picked) >= args.max_per_track:
                break
            if not picked or all(abs(entry["frame"] - p["frame"]) >= MIN_FRAME_GAP for p in picked):
                picked.append(entry)
        for entry in picked:
            fi = entry["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = (int(v) for v in entry["bbox"])
            H, W = frame.shape[:2]
            mx = int((x2 - x1) * MARGIN)
            my = int((y2 - y1) * MARGIN)
            x1c = max(0, x1 - mx)
            y1c = max(0, y1 - my)
            x2c = min(W, x2 + mx)
            y2c = min(H, y2 + my)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            if crop.shape[1] < UPSCALE_TO:
                scale = UPSCALE_TO / crop.shape[1]
                crop = cv2.resize(
                    crop, (UPSCALE_TO, int(crop.shape[0] * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            hue_str = f"{entry['hue']:.0f}" if entry["hue"] is not None else "na"
            tpref = "p" if track_type == "player" else "r"
            name = f"{args.source}_{tpref}{track_id}_f{fi}_h{hue_str}_{category}.png"
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            manifest.append(
                {
                    "file": name,
                    "source": args.source,
                    "track_type": track_type,
                    "track_id": track_id,
                    "frame": fi,
                    "yolo_class": track_type,
                    "median_hue": entry["hue"],
                    "category": category,
                    "reason": f"changed_frames={len(entries)}",
                    "manual_label": None,
                }
            )
        count += 1
    cap.release()

    (out_dir / "candidates.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8",
    )
    counts = defaultdict(int)
    for m in manifest:
        counts[m["category"]] += 1
    print(f"{args.source} verdict: extracted {len(manifest)} crops: {dict(counts)}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
