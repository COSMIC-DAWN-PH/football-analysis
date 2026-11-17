import os
import json
import numpy as np
from PIL import Image

BASE = r"C:\Personal Profile\Profile\Football-Analysis-field-detection"

ITEMS = [
    {
        "id": "demo4_b01723",
        "frame_dir": os.path.join(BASE, "eval", "ball_crops", "demo4", "pilot", "frames"),
        "crop_path": os.path.join(BASE, "eval", "ball_crops", "demo4", "crops", "demo4_b01723.png"),
        "frame_pattern": "demo4_b01723_f{:02d}.jpg",
    },
    {
        "id": "demo4_c00009",
        "frame_dir": os.path.join(BASE, "eval", "ball_crops", "demo4", "pilot", "frames"),
        "crop_path": os.path.join(BASE, "eval", "ball_crops", "demo4", "crops", "demo4_c00009.png"),
        "frame_pattern": "demo4_c00009_f{:02d}.jpg",
    },
    {
        "id": "demo4_b01724",
        "frame_dir": os.path.join(BASE, "eval", "ball_crops", "demo4", "pilot", "frames"),
        "crop_path": os.path.join(BASE, "eval", "ball_crops", "demo4", "crops", "demo4_b01724.png"),
        "frame_pattern": "demo4_b01724_f{:02d}.jpg",
    },
]


def find_red_box(img):
    """Find bounding box of red pixels (R>200, G<80, B<80)."""
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    else:
        return None, 0

    mask = (r > 200) & (g < 80) & (b < 80)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None, 0

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return (int(x_min), int(y_min), int(w), int(h)), (cx, cy)


def analyze_crop(crop_path):
    """Describe crop image contents."""
    img = Image.open(crop_path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    total_pixels = h * w

    # Dominant colors via simple binning
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()

    # Red pixel count
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
        red_mask = (r > 200) & (g < 80) & (b < 80)
        red_pct = red_mask.sum() / total_pixels * 100
    else:
        red_pct = 0

    # Green pixel count (grass-like)
    green_mask = (g > 100) & (r < 120) & (b < 100)
    green_pct = green_mask.sum() / total_pixels * 100

    # White pixel count
    white_mask = (r > 200) & (g > 200) & (b > 200)
    white_pct = white_mask.sum() / total_pixels * 100

    # Check for alpha channel
    has_alpha = arr.ndim == 3 and arr.shape[2] == 4

    desc_parts = []
    desc_parts.append(f"Dimensions: {w}x{h}")
    desc_parts.append(f"Mean RGB: ({r_mean:.0f}, {g_mean:.0f}, {b_mean:.0f})")
    desc_parts.append(f"Red pixels: {red_pct:.1f}%")
    desc_parts.append(f"Green pixels: {green_pct:.1f}%")
    desc_parts.append(f"White pixels: {white_pct:.1f}%")
    if has_alpha:
        desc_parts.append("Has alpha channel")

    # Determine what's visible
    objects = []
    if red_pct > 5:
        objects.append("red ball (bounding box)")
    elif red_pct > 0.1:
        objects.append("small red artifact or partial ball")
    if green_pct > 30:
        objects.append("green field/grass")
    if white_pct > 10:
        objects.append("white markings or background")

    # Look for skin tones (players)
    skin_mask = (r > 180) & (g > 130) & (b < 120) & (r > g) & (g > b)
    skin_pct = skin_mask.sum() / total_pixels * 100
    if skin_pct > 2:
        objects.append(f"possible player skin ({skin_pct:.1f}%)")

    desc_parts.append(f"Visible: {', '.join(objects)}")
    return "\n".join(desc_parts)


def classify_movement(centers, red_counts):
    """Determine if object moves across frames."""
    valid = [(i, c) for i, (c, rc) in enumerate(zip(centers, red_counts)) if c is not None and rc > 0]
    if len(valid) == 0:
        return "no_red_detected", centers, red_counts

    frame_indices = [v[0] for v in valid]
    frame_centers = [v[1] for v in valid]

    if len(frame_centers) <= 1:
        return "single_frame", centers, red_counts

    # Compute max displacement between consecutive frames
    max_disp = 0
    for i in range(1, len(frame_centers)):
        dx = frame_centers[i][0] - frame_centers[i-1][0]
        dy = frame_centers[i][1] - frame_centers[i-1][1]
        disp = np.sqrt(dx**2 + dy**2)
        max_disp = max(max_disp, disp)

    avg_red = sum(red_counts[i] for i in frame_indices) / len(frame_indices)

    return max_disp, centers, red_counts


print("=" * 80)
print("FOOTBALL BALL DETECTION ANALYSIS - Items 4, 5, 6")
print("=" * 80)

results = {}

for item in ITEMS:
    item_id = item["id"]
    print(f"\n{'='*80}")
    print(f"ITEM: {item_id}")
    print(f"{'='*80}")

    frame_paths = [os.path.join(item["frame_dir"], item["frame_pattern"].format(i)) for i in range(1, 16)]
    centers = []
    boxes = []
    red_counts_list = []

    for fp in frame_paths:
        fname = os.path.basename(fp)
        if not os.path.exists(fp):
            print(f"  [{fname}] FILE NOT FOUND")
            centers.append(None)
            boxes.append(None)
            red_counts_list.append(0)
            continue

        img = Image.open(fp).convert("RGB")
        box, center = find_red_box(img)
        arr = np.array(img)
        r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
        red_mask = (r > 200) & (g < 80) & (b < 80)
        red_count = int(red_mask.sum())

        centers.append(center)
        boxes.append(box)
        red_counts_list.append(red_count)

        status = ""
        if box:
            status = f"BOX: ({box[0]}, {box[1]}, {box[2]}x{box[3]}) CENTER: ({center[0]:.1f}, {center[1]:.1f}) RED_PIXELS: {red_count}"
        else:
            status = f"NO_RED_BOX  RED_PIXELS: {red_count}"
        print(f"  [{fname}] {status}")

    # Analyze crop
    crop_desc = analyze_crop(item["crop_path"])
    print(f"\n  CROP IMAGE ({item['crop_path']}):")
    for line in crop_desc.split("\n"):
        print(f"    {line}")

    # Movement analysis
    print(f"\n  MOVEMENT ANALYSIS:")
    result = classify_movement(centers, red_counts_list)

    classification = None
    reason = ""

    if isinstance(result, tuple) and result[0] == "no_red_detected":
        classification = "not_ball"
        reason = "No red bounding box detected in any frame - no ball present"
        print(f"    Classification: NOT_BALL - No red bounding box detected in any frame")

    elif isinstance(result, tuple) and result[0] == "single_frame":
        classification = "not_ball"
        reason = "Red box detected in only a single frame - a rolling ball would appear in multiple consecutive frames"
        print(f"    Classification: NOT_BALL - Detected in only one frame")

    else:
        max_disp = result[0]
        avg_red = sum(red_counts_list[i] for i, c in enumerate(centers) if c is not None) / max(sum(1 for c in centers if c is not None), 1)
        total_red = sum(red_counts_list)

        print(f"    Max displacement: {max_disp:.1f} pixels")
        print(f"    Average red pixels per frame: {avg_red:.0f}")
        print(f"    Total red pixels across all frames: {total_red}")

        valid_frames = sum(1 for c in centers if c is not None)
        print(f"    Frames with red detection: {valid_frames}/15")

        if valid_frames == 0:
            classification = "not_ball"
            reason = "No red bounding box detected in any frame"
            print(f"    Classification: NOT_BALL - No red detected")

        elif valid_frames < 3:
            classification = "not_ball"
            reason = f"Red box detected in only {valid_frames} frames (out of 15) - insufficient continuity for a rolling ball"
            print(f"    Classification: NOT_BALL - Only {valid_frames} frames with detection")

        elif max_disp < 20:
            classification = "not_ball"
            reason = f"Red box barely moved (max {max_disp:.1f}px) - likely a static red marking/artifact, not a rolling ball"
            print(f"    Classification: NOT_BALL - Minimal movement ({max_disp:.1f}px)")

        elif max_disp < 100 and avg_red > 100:
            classification = "ball"
            reason = f"Red box moves significantly ({max_disp:.1f}px max displacement) with substantial red pixels ({avg_red:.0f}/frame) - consistent with a rolling ball"
            print(f"    Classification: BALL - Significant movement with good red signal")

        elif max_disp >= 100 and avg_red > 200:
            classification = "ball"
            reason = f"Red box moves substantially ({max_disp:.1f}px) with strong red signal ({avg_red:.0f}px/frame) - clear ball trajectory"
            print(f"    Classification: BALL - Strong movement + strong red signal")

        else:
            classification = "not_ball"
            reason = f"Weak red signal ({avg_red:.0f} px/frame) despite movement ({max_disp:.1f}px) - likely false positive, not a solid ball"
            print(f"    Classification: NOT_BALL - Weak red signal despite movement")

    print(f"\n  FINAL LABEL: {classification}")
    print(f"  REASON: {reason}")

    results[item_id] = {"label": classification, "reason": reason}

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
for item_id, data in results.items():
    print(f"  {item_id}: {data['label']} - {data['reason']}")

print(f"\nJSON output:")
print(json.dumps(results, indent=2))
