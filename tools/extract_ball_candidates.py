"""Extract ball-review candidate crops from ball_tracks.jsonl + source video.

Review scope (plan/BALL_DETECTION.md section 5) - three sources of crops:

  confirmed     tracker segments (observed frames), <=3 clearest frames per
                segment, frame gap >= 15; crop = track bbox + context margin
  unconfirmed   raw candidates never accepted by the tracker (IoU<0.3 vs the
                observed track bbox of that frame), greedy spatio-temporal
                clustered; one crop per cluster = FP-prone material
  bridge_sweep  predicted (observed=False) frames inside segments: detector
                missed the ball but the tracker bridged it - FN material,
                <=2 crops per segment, wide context window
  gap_sweep     midpoint frame between consecutive segments (gap<=5s): where
                the ball was lost between segments, wide context window
  global_sweep  blank frames (no candidates) sampled every N frames: blind
                FN discovery, full-frame downscaled crop

Every item is written to out/candidates.jsonl with manual_label/qwen_label/
luna_label pre-seeded to null. The human reviews EVERY crop; the dual models
(qwen3.7plus / gpt5.6luna) prefill their own label files which are merged
back in by tools/prefill_ball_labels.py.

Usage:
  python tools/extract_ball_candidates.py \
      --video C:/Personal/Profile/Video/demo4.mp4 \
      --tracks output_videos/demo4/ball/ball_tracks.jsonl \
      --source demo4 --out eval/ball_crops/demo4
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

MAX_PER_SEGMENT = 3
MIN_FRAME_GAP = 15
IOU_UNTRACKED = 0.3
CLUSTER_DIST_FRAC = 0.03
CLUSTER_GAP_FRAMES = 15
BRIDGE_PER_SEGMENT = 2
GAP_MAX_FRAMES = 150
GLOBAL_STRIDE = 300
GLOBAL_OFFSET = 150
FULL_FRAME_MAX_DIM = 960
CONTEXT_MARGIN_FRAC = 0.8
WIDE_CONTEXT_MARGIN_FRAC = 2.0
MIN_CONTEXT_PX = 12.0
MIN_SMALL_DIM = 96


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _crop_box(bbox, margin_px, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1 - margin_px)); y1 = max(0, int(y1 - margin_px))
    x2 = min(frame_w, int(x2 + margin_px)); y2 = min(frame_h, int(y2 + margin_px))
    return x1, y1, x2, y2


def _save_crop(frame, crop_box, target_dim, out_path, draw_bbox=None):
    x1, y1, x2, y2 = crop_box
    tile = frame[y1:y2, x1:x2]
    if tile.size == 0:
        return None
    h, w = tile.shape[:2]
    longest = max(h, w)
    if longest > 0:
        scale = target_dim / longest
        if abs(scale - 1.0) > 0.02:
            tile = cv2.resize(
                tile, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
            )
    if draw_bbox is not None:
        bx = draw_bbox
        bx = [bx[0] - x1, bx[1] - y1, bx[2] - x1, bx[3] - y1]
        if scale != 1.0:
            bx = [int(v * scale) for v in bx]
        color = (0, 0, 255)
        cv2.rectangle(tile, (bx[0], bx[1]), (bx[2], bx[3]), color, 1)
    cv2.imwrite(str(out_path), tile)
    return tile.shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--source", required=True, help="e.g. demo4 / raw1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-segment", type=int, default=MAX_PER_SEGMENT)
    parser.add_argument("--global-stride", type=int, default=GLOBAL_STRIDE)
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Missing video: {args.video}")
    if not args.tracks.is_file():
        raise FileNotFoundError(f"Missing tracks: {args.tracks}")

    cap = cv2.VideoCapture(str(args.video))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diagonal = (frame_w * frame_w + frame_h * frame_h) ** 0.5
    cluster_dist = diagonal * CLUSTER_DIST_FRAC

    records = [json.loads(line) for line in args.tracks.read_text(encoding="utf-8").splitlines()]

    # --- Pass A: decide review items -------------------------------------------------
    segments = defaultdict(list)          # segment -> [frame records]
    segment_frames = {}                   # segment -> (first_frame, last_frame)
    for d in records:
        track = d["track"]
        if track and track["track_segment"]:
            segments[track["track_segment"]].append(d)
            first, last = segment_frames.get(
                track["track_segment"], (d["frame"], d["frame"]))
            segment_frames[track["track_segment"]] = (
                min(first, d["frame"]), max(last, d["frame"]))

    items = []          # dicts describing crops
    next_id = [0]
    def new_id(category):
        next_id[0] += 1
        return f"{args.source}_{category[0]}{next_id[0]:05d}"

    # confirmed: clearest observed frames per segment
    for seg, frames in sorted(segments.items()):
        observed = [d for d in frames if d["track"]["observed"]]
        observed.sort(key=lambda d: -d["track"]["track_confidence"])
        picked = []
        for d in observed:
            if len(picked) >= args.max_per_segment:
                break
            if all(abs(d["frame"] - p["frame"]) >= MIN_FRAME_GAP for p in picked):
                picked.append(d)
        for d in picked:
            bbox = d["track"]["bbox"]
            margin = max(
                MIN_CONTEXT_PX,
                max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * CONTEXT_MARGIN_FRAC,
            )
            items.append({
                "id": new_id("confirmed"), "src": args.source, "frame": d["frame"],
                "t": d["t"], "category": "confirmed", "segment": seg,
                "conf": d["track"]["confidence"], "bbox": bbox,
                "cluster": None, "crop": None, "margin": margin,
                "target": None,
                "features": {"bbox_w": round(bbox[2] - bbox[0], 1),
                             "bbox_h": round(bbox[3] - bbox[1], 1)},
            })

    # unconfirmed: clustered raw candidates not covered by the observed track
    untracked = []  # (frame, t, candidate, conf)
    for d in records:
        track_bbox = d["track"]["bbox"] if d["track"] and d["track"]["observed"] else None
        for cand in d["candidates"]:
            x1, y1, x2, y2, conf = cand
            if track_bbox and _iou([x1, y1, x2, y2], track_bbox) >= IOU_UNTRACKED:
                continue
            untracked.append((d["frame"], d["t"], [x1, y1, x2, y2], conf))
    untracked.sort(key=lambda c: -c[3])
    clusters = []   # list of [members]
    for frame, t, bbox, conf in untracked:
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        placed = False
        for cluster in clusters:
            last_frame = max(m[0] for m in cluster)
            if abs(frame - last_frame) <= CLUSTER_GAP_FRAMES:
                for m in cluster:
                    mx, my = (m[2][0] + m[2][2]) / 2, (m[2][1] + m[2][3]) / 2
                    if ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5 <= cluster_dist:
                        cluster.append((frame, t, bbox, conf))
                        placed = True
                        break
            if placed:
                break
        if not placed:
            clusters.append([(frame, t, bbox, conf)])
    for ci, cluster in enumerate(clusters):
        best = max(cluster, key=lambda m: m[3])
        frame, t, bbox, conf = best
        margin = max(
            MIN_CONTEXT_PX,
            max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * CONTEXT_MARGIN_FRAC,
        )
        items.append({
            "id": new_id("unconfirmed"), "src": args.source, "frame": frame,
            "t": t, "category": "unconfirmed", "segment": None, "conf": conf,
            "bbox": bbox, "cluster": ci, "crop": None, "margin": margin,
            "target": None,
            "features": {"bbox_w": round(bbox[2] - bbox[0], 1),
                         "bbox_h": round(bbox[3] - bbox[1], 1),
                         "cluster_size": len(cluster),
                         "cluster_frames": len({m[0] for m in cluster})},
        })

    # bridge_sweep: predicted frames inside segments
    for seg, frames in sorted(segments.items()):
        predicted = [d for d in frames if not d["track"]["observed"]]
        if not predicted:
            continue
        picked = predicted[: args.max_per_segment if args.max_per_segment >= 2 else BRIDGE_PER_SEGMENT]
        if len(predicted) > (args.max_per_segment if args.max_per_segment >= 2 else BRIDGE_PER_SEGMENT):
            step = max(1, len(predicted) // 2)
            picked = predicted[::step][: (args.max_per_segment if args.max_per_segment >= 2 else BRIDGE_PER_SEGMENT)]
        for d in picked:
            bbox = d["track"]["bbox"]
            margin = max(
                MIN_CONTEXT_PX,
                max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * WIDE_CONTEXT_MARGIN_FRAC,
            )
            items.append({
                "id": new_id("bridge"), "src": args.source, "frame": d["frame"],
                "t": d["t"], "category": "bridge_sweep", "segment": seg,
                "conf": d["track"]["confidence"], "bbox": bbox, "cluster": None,
                "crop": None, "margin": margin, "target": None,
                "features": {"bbox_w": round(bbox[2] - bbox[0], 1),
                             "bbox_h": round(bbox[3] - bbox[1], 1)},
            })

    # gap_sweep: midpoint between consecutive segments
    sorted_segs = sorted(segment_frames.items())
    for (seg_a, (fa, la)), (seg_b, (fb, lb)) in zip(sorted_segs, sorted_segs[1:]):
        if seg_b == seg_a:
            continue
        if 0 < fb - la <= GAP_MAX_FRAMES:
            mid = (la + fb) // 2
            d = records[mid]
            track = d["track"]
            bbox = track["bbox"] if track else None
            if bbox is None:
                bbox = [frame_w * 0.4, frame_h * 0.3, frame_w * 0.6, frame_h * 0.7]
                target = FULL_FRAME_MAX_DIM
                margin = 0.0
            else:
                target = None
                margin = max(
                    MIN_CONTEXT_PX,
                    max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * WIDE_CONTEXT_MARGIN_FRAC,
                )
            items.append({
                "id": new_id("gap"), "src": args.source, "frame": mid,
                "t": d["t"], "category": "gap_sweep",
                "segment": f"{seg_a}>{seg_b}",
                "conf": track["confidence"] if track else None, "bbox": bbox,
                "cluster": None, "crop": None, "margin": margin,
                "target": target, "features": {"gap_frames": fb - la},
            })

    # global_sweep: blank frames sampled on a fixed stride
    blank_frames = [d["frame"] for d in records if not d["candidates"]]
    picked_global = []
    for frame in blank_frames:
        if (frame - GLOBAL_OFFSET) % args.global_stride == 0:
            picked_global.append(frame)
    for frame in picked_global:
        d = records[frame]
        bbox = [0, 0, frame_w, frame_h]
        items.append({
            "id": new_id("global"), "src": args.source, "frame": frame,
            "t": d["t"], "category": "global_sweep", "segment": None,
            "conf": None, "bbox": bbox, "cluster": None, "crop": None,
            "margin": 0.0, "target": FULL_FRAME_MAX_DIM,
            "features": {"blank_sampled": True},
        })

    # --- Pass B: sequential video read and crop saving ------------------------------
    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    by_frame = defaultdict(list)
    for item in items:
        by_frame[item["frame"]].append(item)

    needed = set(by_frame.keys())
    saved = {}
    frame_idx = 0
    while needed:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            for item in by_frame[frame_idx]:
                box = _crop_box(item["bbox"], item["margin"], frame_w, frame_h)
                rel = f"crops/{item['id']}.png"
                if item["target"] is not None:
                    shape = _save_crop(frame, box, item["target"], args.out / rel)
                else:
                    longest = max(box[2] - box[0], box[3] - box[1])
                    target = max(MIN_SMALL_DIM, longest)
                    shape = _save_crop(
                        frame, box, target, args.out / rel, draw_bbox=item["bbox"])
                item["crop"] = rel
                if shape is not None:
                    item["features"]["crop_w"], item["features"]["crop_h"] = shape[0], shape[1]
                saved[item["id"]] = item
            needed.discard(frame_idx)
        frame_idx += 1
    cap.release()

    missing = [i["id"] for i in items if i["id"] not in saved]
    if missing:
        print(f"WARNING: {len(missing)} crops could not be saved (missing frames)")

    manifest = []
    for item in items:
        if item["id"] not in saved:
            continue
        entry = {
            "id": item["id"], "src": item["src"], "frame": item["frame"],
            "t": item["t"], "category": item["category"], "segment": item["segment"],
            "conf": item["conf"], "bbox": item["bbox"], "cluster": item["cluster"],
            "crop": item["crop"], "features": item["features"],
            "manual_label": None, "qwen_label": None, "luna_label": None,
        }
        manifest.append(entry)

    out_path = args.out / "candidates.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in manifest) + "\n",
        encoding="utf-8",
    )
    from collections import Counter
    counts = Counter(e["category"] for e in manifest)
    print(f"{args.source}: {len(manifest)} review items -> {out_path}")
    print(f"  categories: {dict(counts)}")
    print(f"  segments: {len(segments)}, untracked clusters: {len(clusters)}, "
          f"blank frames: {len(blank_frames)}/{len(records)}")


if __name__ == "__main__":
    main()
