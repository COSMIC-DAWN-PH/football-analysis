import os
import json
import numpy as np
from PIL import Image
from collections import OrderedDict

BASE = "C:\\Personal Profile\\Profile\\Football-Analysis-field-detection"
ITEMS = [
    {
        "id": "demo4_c00012",
        "frame_pattern": os.path.join(BASE, "eval", "ball_crops", "demo4", "pilot", "frames", "demo4_c00012_f{:02d}.jpg"),
        "crop_path": os.path.join(BASE, "eval", "ball_crops", "demo4", "crops", "demo4_c00012.png"),
    },
    {
        "id": "demo4_b01725",
        "frame_pattern": os.path.join(BASE, "eval", "ball_crops", "demo4", "pilot", "frames", "demo4_b01725_f{:02d}.jpg"),
        "crop_path": os.path.join(BASE, "eval", "ball_crops", "demo4", "crops", "demo4_b01725.png"),
    },
    {
        "id": "demo4_c00015",
        "frame_pattern": os.path.join(BASE, "eval", "ball_crops", "demo4", "pilot", "frames", "demo4_c00015_f{:02d}.jpg"),
        "crop_path": os.path.join(BASE, "eval", "ball_crops", "demo4", "crops", "demo4_c00015.png"),
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
    red_count = int(mask.sum())
    if red_count == 0:
        return None, 0
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "cx": center_x, "cy": center_y, "red_pixels": red_count}, red_count


def analyze_crop(crop_path):
    """Describe crop image contents."""
    img = Image.open(crop_path)
    w, h = img.size
    arr = np.array(img)

    # Dominant colors via simple binning
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r_mean = float(arr[:, :, 0].mean())
        g_mean = float(arr[:, :, 1].mean())
        b_mean = float(arr[:, :, 2].mean())
    elif arr.ndim == 2:
        r_mean = g_mean = b_mean = float(arr.mean())
    else:
        r_mean = g_mean = b_mean = 0.0

    # Check for red content in crop
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
        red_mask = (r > 200) & (g < 80) & (b < 80)
        green_mask = (g > 100) & (r < 100) & (b < 100)
        red_pct = float(red_mask.sum()) / float(arr.shape[0] * arr.shape[1]) * 100
        green_pct = float(green_mask.sum()) / float(arr.shape[0] * arr.shape[1]) * 100
    else:
        red_pct = green_pct = 0.0

    # Check alpha channel
    has_alpha = arr.ndim == 3 and arr.shape[2] == 4
    alpha_info = ""
    if has_alpha:
        alpha_arr = arr[:, :, 3]
        transparent = (alpha_arr < 128).sum()
        opaque = (alpha_arr >= 200).sum()
        total = alpha_arr.size
        alpha_info = f", alpha: {transparent/total*100:.1f}% transparent, {opaque/total*100:.1f}% opaque"

    desc = f"Crop: {w}x{h}, dominant RGB=({r_mean:.0f},{g_mean:.0f},{b_mean:.0f}), "
    desc += f"red={red_pct:.1f}%, green={green_pct:.1f}%{alpha_info}"

    # Try to identify what's visible
    objects = []
    if red_pct > 5:
        objects.append("contains significant red region")
    if green_pct > 20:
        objects.append("predominantly green (grass)")
    elif green_pct > 5:
        objects.append("some green areas")
    if red_pct < 1 and green_pct < 5:
        objects.append("no distinct red or green objects visible")
    if not objects:
        objects.append("mixed content")

    return desc, objects


def compute_movement(frame_results):
    """Analyze whether red box center moves across frames."""
    centers = [(r["cx"], r["cy"]) for r in frame_results if r is not None]
    if len(centers) < 2:
        return "insufficient_data"
    dx_list = []
    dy_list = []
    for i in range(1, len(centers)):
        dx_list.append(abs(centers[i][0] - centers[i-1][0]))
        dy_list.append(abs(centers[i][1] - centers[i-1][1]))
    avg_dx = float(np.mean(dx_list))
    avg_dy = float(np.mean(dy_list))
    max_dx = float(np.max(dx_list))
    max_dy = float(np.max(dy_list))
    total_dx = abs(centers[-1][0] - centers[0][0])
    total_dy = abs(centers[-1][1] - centers[0][1])
    return {
        "n_frames_with_red": len(centers),
        "avg_frame_to_frame_dx": round(avg_dx, 2),
        "avg_frame_to_frame_dy": round(avg_dy, 2),
        "max_frame_to_frame_dx": round(max_dx, 2),
        "max_frame_to_frame_dy": round(max_dy, 2),
        "total_displacement_px": round((total_dx**2 + total_dy**2)**0.5, 2),
        "total_dx": round(total_dx, 2),
        "total_dy": round(total_dy, 2),
    }


def classify(movement, crop_desc, crop_objects, frame_details):
    """Classify as ball, not_ball, or null."""
    n_with_red = movement["n_frames_with_red"]
    total = 15
    if n_with_red == 0:
        return "not_ball", "No red bounding box detected in any frame"

    if n_with_red < 3:
        return "not_ball", f"Red box detected in only {n_with_red}/{total} frames — too sparse for a ball track"

    # Check if red box exists in ALL frames (a tracked ball should appear consistently)
    if n_with_red < total:
        missing = total - n_with_red
        if missing > 5:
            return "not_ball", f"Red box missing in {missing}/{total} frames — inconsistent tracking"

    avg_move = (movement["avg_frame_to_frame_dx"]**2 + movement["avg_frame_to_frame_dy"]**2)**0.5
    total_disp = movement["total_displacement_px"]

    # A rolling ball should move smoothly and continuously
    if avg_move > 50:
        return "ball", f"Strong continuous motion detected: avg {avg_move:.1f}px/frame, total displacement {total_disp:.1f}px — consistent with rolling ball"
    elif avg_move > 10:
        return "ball", f"Moderate continuous motion: avg {avg_move:.1f}px/frame, total displacement {total_disp:.1f}px — consistent with rolling/slow-moving ball"
    elif avg_move > 2:
        return "ball", f"Subtle but consistent motion: avg {avg_move:.1f}px/frame — likely a ball moving slowly or camera panning"
    else:
        # Very little movement could be ball on ground or static marker
        # Check crop for context
        crop_str = crop_desc.lower()
        if "grass" in crop_str or "green" in crop_str or "green=" in crop_str:
            return "ball", f"Minimal motion ({avg_move:.1f}px/frame avg) on grass field — ball may be stationary or very slow; red marker present in all frames"
        else:
            return "not_ball", f"Red box present but minimal motion ({avg_move:.1f}px/frame) without clear grass/field context — may be a static red marker"


results = []

for item in ITEMS:
    print(f"\n{'='*80}")
    print(f"ITEM: {item['id']}")
    print(f"{'='*80}")

    frame_results = []
    for f_num in range(1, 16):
        frame_path = item["frame_pattern"].format(f_num)
        try:
            img = Image.open(frame_path).convert("RGB")
            w, h = img.size
            box, red_count = find_red_box(img)
            entry = {
                "frame": f"f{f_num:02d}",
                "path": frame_path,
                "dims": f"{w}x{h}",
                "box": box,
                "red_pixels": red_count,
            }
            frame_results.append(entry)
            if box:
                print(f"  {entry['frame']}: RED BOX found at ({box['x_min']},{box['y_min']})-({box['x_max']},{box['y_max']}), center=({box['cx']:.1f},{box['cy']:.1f}), red_px={red_count}")
            else:
                print(f"  {entry['frame']}: No red box detected (red_px={red_count})")
        except Exception as e:
            print(f"  {entry['frame']}: ERROR - {e}")
            frame_results.append(None)

    # Movement analysis
    valid_results = [r for r in frame_results if r is not None and r["box"] is not None]
    n_valid = len(valid_results)
    movement_data = {"n_frames_with_red": n_valid, "avg_frame_to_frame_dx": 0, "avg_frame_to_frame_dy": 0, "max_frame_to_frame_dx": 0, "max_frame_to_frame_dy": 0, "total_displacement_px": 0, "total_dx": 0, "total_dy": 0}
    if n_valid >= 2:
        movement_data = compute_movement([r["box"] for r in valid_results])
    print(f"\n  MOVEMENT ANALYSIS:")
    print(f"    Frames with red box: {movement_data['n_frames_with_red']}/15")
    if n_valid >= 2:
        print(f"    Avg frame-to-frame shift: {movement_data['avg_frame_to_frame_dx']:.1f}px (X), {movement_data['avg_frame_to_frame_dy']:.1f}px (Y)")
        print(f"    Max single-frame shift: {movement_data['max_frame_to_frame_dx']:.1f}px (X), {movement_data['max_frame_to_frame_dy']:.1f}px (Y)")
        print(f"    Total displacement (start->end): {movement_data['total_displacement_px']:.1f}px")
    elif n_valid == 1:
        print(f"    Only 1 frame has red box — cannot measure motion trajectory")
        print(f"    Red box location: ({valid_results[0]['box']['x_min']},{valid_results[0]['box']['y_min']})-({valid_results[0]['box']['x_max']},{valid_results[0]['box']['y_max']}), center=({valid_results[0]['box']['cx']:.1f},{valid_results[0]['box']['cy']:.1f}), red_px={valid_results[0]['box']['red_pixels']}")
    else:
        print(f"    No red boxes found in any frame")

    # Crop analysis
    print(f"\n  CROP ANALYSIS:")
    crop_desc, crop_objects = analyze_crop(item["crop_path"])
    print(f"    {crop_desc}")
    for obj in crop_objects:
        print(f"    -> {obj}")

    # Classification
    label, reason = classify(movement_data, crop_desc, crop_objects, frame_results)
    print(f"\n  CLASSIFICATION: {label.upper()}")
    print(f"  Reason: {reason}")

    results.append({
        "id": item["id"],
        "label": label,
        "reason": reason,
        "movement": movement_data,
        "crop_description": crop_desc,
        "crop_objects": crop_objects,
        "frames_detected": movement_data["n_frames_with_red"],
    })

# Summary JSON
print(f"\n\n{'='*80}")
print("FINAL SUMMARY JSON")
print(f"{'='*80}")
summary = [{"id": r["id"], "label": r["label"], "reason": r["reason"]} for r in results]
print(json.dumps(summary, indent=2))
