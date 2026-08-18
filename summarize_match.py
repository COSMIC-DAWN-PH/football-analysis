"""Create quality-filtered tactical summaries from per-frame JSONL tracks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from position_mappers import PitchGeometry


TEAM_COLORS = {"Red": "#d82f45", "Blue": "#2764c7"}


def default_summary_dir(tracks_dir: Path) -> Path:
    """Place summaries beside a conventional raw track directory."""
    if tracks_dir.name.casefold() == "raw":
        return tracks_dir.parent / "summary"
    return tracks_dir / "summary"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Summary directory; defaults to a summary folder beside raw tracks",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--club1-name", default="Red",
                        help="Club name stored in object_tracks.jsonl; maps to the red team in charts")
    parser.add_argument("--club2-name", default="Blue",
                        help="Club name stored in object_tracks.jsonl; maps to the blue team in charts")
    parser.add_argument("--pitch-length-m", type=float, required=True)
    parser.add_argument("--pitch-width-m", type=float, required=True)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = default_summary_dir(args.tracks_dir)
    return args


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_homography(
    keypoints: dict, geometry: PitchGeometry
) -> tuple[np.ndarray | None, dict]:
    pairs = []
    for raw_idx, raw_xy in keypoints.items():
        idx = int(raw_idx)
        if 0 <= idx < len(geometry.vertices):
            pairs.append((np.asarray(raw_xy, dtype=np.float32), geometry.vertices[idx]))

    quality = {
        "keypoints": len(pairs),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_error": None,
        "pitch_span_x": 0.0,
        "pitch_span_y": 0.0,
    }
    if len(pairs) < 6:
        return None, quality

    src = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
    dst = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
    world_to_image, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    if world_to_image is None or mask is None:
        return None, quality

    mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), world_to_image).reshape(-1, 2)
    errors = np.linalg.norm(projected - src, axis=1)
    inliers = int(mask.sum())
    median_error = float(np.median(errors[mask])) if inliers else float("inf")
    inlier_ratio = inliers / len(pairs)
    pitch_span_x = float(np.ptp(dst[mask, 0])) if inliers else 0.0
    pitch_span_y = float(np.ptp(dst[mask, 1])) if inliers else 0.0
    quality.update(
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        median_error=median_error,
        pitch_span_x=pitch_span_x,
        pitch_span_y=pitch_span_y,
    )

    # Match the online calibrator's acceptance gates so legacy JSONL files do
    # not silently admit lower-quality projections than current runs.
    if (
        inliers < 5
        or inlier_ratio < 0.60
        or median_error > 5.0
        or pitch_span_x / geometry.length_m < 0.30
        or pitch_span_y / geometry.width_m < 0.25
    ):
        return None, quality
    try:
        return np.linalg.inv(world_to_image), quality
    except np.linalg.LinAlgError:
        return None, quality


def project_players(
    objects: dict,
    homography: np.ndarray | None,
    geometry: PitchGeometry,
    team_names: tuple[str, str] = ("Red", "Blue"),
) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {name: [] for name in team_names}
    for object_type in ("player", "goalkeeper"):
        for item in objects.get(object_type, {}).values():
            team = item.get("club")
            if team not in result:
                continue
            position = item.get("position_m")
            if position is None and homography is not None:
                x1, _y1, x2, y2 = item["bbox"]
                foot = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float32)
                position = cv2.perspectiveTransform(foot, homography).reshape(2)
            if position is not None and geometry.contains(tuple(position), margin_m=0.0):
                result[team].append((float(position[0]), float(position[1])))
    return result


def draw_pitch(axis: plt.Axes, geometry: PitchGeometry) -> None:
    axis.set_facecolor("#1f6b45")
    line = "#e8f0e8"
    length, width = geometry.length_m, geometry.width_m
    axis.plot([0, length, length, 0, 0], [0, 0, width, width, 0], color=line, lw=1.2)
    axis.plot([length / 2, length / 2], [0, width], color=line, lw=1.0)
    axis.add_patch(
        plt.Circle(
            (length / 2, width / 2),
            geometry.centre_circle_radius_m,
            fill=False,
            color=line,
            lw=1.0,
        )
    )
    penalty_y = (width - geometry.penalty_area_width_m) / 2
    goal_y = (width - geometry.goal_area_width_m) / 2
    for x, sign in ((0, 1), (length, -1)):
        axis.add_patch(
            plt.Rectangle(
                (x, penalty_y),
                sign * geometry.penalty_area_depth_m,
                geometry.penalty_area_width_m,
                fill=False,
                color=line,
                lw=1.0,
            )
        )
        axis.add_patch(
            plt.Rectangle(
                (x, goal_y),
                sign * geometry.goal_area_depth_m,
                geometry.goal_area_width_m,
                fill=False,
                color=line,
                lw=1.0,
            )
        )
    axis.set_xlim(0, length)
    axis.set_ylim(width, 0)
    axis.set_aspect("equal")
    axis.set_xlabel("Fixed pitch x (m)")
    axis.set_ylabel("Fixed pitch y (m)")


def smooth_histogram(
    points: list[tuple[float, float]], geometry: PitchGeometry
) -> np.ndarray:
    if not points:
        return np.zeros((68, 105), dtype=np.float32)
    values = np.asarray(points)
    histogram, _, _ = np.histogram2d(
        values[:, 1],
        values[:, 0],
        bins=(68, 105),
        range=((0, geometry.width_m), (0, geometry.length_m)),
    )
    return cv2.GaussianBlur(histogram.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)


def per_frame_metrics(points: list[tuple[float, float]]) -> dict | None:
    if not points:
        return None
    values = np.asarray(points)
    result = {"visible": len(points)}
    if len(points) >= 2:
        centroid = values.mean(axis=0)
        result.update(
            centroid_x=float(centroid[0]),
            centroid_y=float(centroid[1]),
        )
    if len(points) >= 3:
        result.update(
            length=float(np.percentile(values[:, 0], 90) - np.percentile(values[:, 0], 10)),
            width=float(np.percentile(values[:, 1], 90) - np.percentile(values[:, 1], 10)),
            compactness=float(np.linalg.norm(values - values.mean(axis=0), axis=1).mean()),
        )
    return result


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def fmt(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.1f}{suffix}"


def main() -> None:
    args = parse_args()
    geometry = PitchGeometry(args.pitch_length_m, args.pitch_width_m)
    team_names = (args.club1_name, args.club2_name)
    colors = dict(TEAM_COLORS)
    colors.setdefault(team_names[0], "#d82f45")
    colors.setdefault(team_names[1], "#2764c7")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = read_jsonl(args.tracks_dir / "object_tracks.jsonl")
    keypoints = read_jsonl(args.tracks_dir / "keypoint_tracks.jsonl")
    frame_count = min(len(objects), len(keypoints))
    calibration_path = args.tracks_dir / "calibration_tracks.jsonl"
    calibrations = read_jsonl(calibration_path) if calibration_path.is_file() else []
    if calibrations:
        frame_count = min(frame_count, len(calibrations))

    all_points: dict[str, list[tuple[float, float]]] = {name: [] for name in team_names}
    frame_rows: list[dict] = []
    quality_rows: list[dict] = []

    for frame_idx in range(frame_count):
        frame_objects = objects[frame_idx]
        frame_keypoints = keypoints[frame_idx]
        calibration = calibrations[frame_idx] if calibrations else None
        has_metric_positions = any(
            "position_m" in item
            for object_type in ("player", "goalkeeper")
            for item in frame_objects.get(object_type, {}).values()
        )
        if calibration is not None:
            calibration_quality = calibration.get("quality")
            accepted_frame = calibration.get("image_to_pitch") is not None and (
                calibration_quality is None or float(calibration_quality) > 0.0
            )
            quality = {
                "keypoints": calibration.get("keypoints", 0),
                "inliers": calibration.get("inliers", 0),
                "inlier_ratio": calibration.get("inlier_ratio", 0.0),
                "median_error": calibration.get("median_error_px"),
                "pitch_span_x": calibration.get("span_length_ratio", 0.0) * geometry.length_m,
                "pitch_span_y": calibration.get("span_width_ratio", 0.0) * geometry.width_m,
                "status": calibration.get("status"),
                "quality": calibration_quality,
                "age_seconds": calibration.get("age_seconds"),
                "flow_quality": calibration.get("flow_quality"),
            }
            homography = None
        else:
            homography, quality = frame_homography(frame_keypoints, geometry)
            accepted_frame = homography is not None
        quality_rows.append({"frame": frame_idx, **quality, "accepted": accepted_frame})
        if not accepted_frame or (not has_metric_positions and homography is None):
            continue
        projected = project_players(frame_objects, homography, geometry, team_names)
        timestamp = (
            float(calibration.get("timestamp_seconds", frame_idx / args.fps))
            if calibration is not None
            else frame_idx / args.fps
        )
        row = {"frame": frame_idx, "second": timestamp}
        usable = False
        for team in team_names:
            all_points[team].extend(projected[team])
            metrics = per_frame_metrics(projected[team])
            if metrics is not None:
                usable = True
                row.update({f"{team.lower()}_{key}": value for key, value in metrics.items()})
        prefix1, prefix2 = (name.lower() for name in team_names)
        if f"{prefix1}_centroid_x" in row and f"{prefix2}_centroid_x" in row:
            row["centroid_separation"] = abs(
                row[f"{prefix1}_centroid_x"] - row[f"{prefix2}_centroid_x"]
            )
        if usable:
            frame_rows.append(row)

    accepted = sum(row["accepted"] for row in quality_rows)
    usable_both = sum(
        f"{prefix1}_centroid_x" in row and f"{prefix2}_centroid_x" in row
        for row in frame_rows
    )
    accepted_quality = [
        float(row["quality"])
        for row in quality_rows
        if row["accepted"] and row.get("quality") is not None
    ]

    team_summary = {}
    for team in team_names:
        prefix = team.lower()
        team_rows = [row for row in frame_rows if f"{prefix}_centroid_x" in row]
        points = np.asarray(all_points[team], dtype=float)
        thirds_x = [0.0, 0.0, 0.0]
        lanes_y = [0.0, 0.0, 0.0]
        if len(points):
            thirds_x = [float(np.mean(points[:, 0] < geometry.length_m / 3)),
                        float(np.mean((points[:, 0] >= geometry.length_m / 3) & (points[:, 0] < 2 * geometry.length_m / 3))),
                        float(np.mean(points[:, 0] >= 2 * geometry.length_m / 3))]
            lanes_y = [float(np.mean(points[:, 1] < geometry.width_m / 3)),
                       float(np.mean((points[:, 1] >= geometry.width_m / 3) & (points[:, 1] < 2 * geometry.width_m / 3))),
                       float(np.mean(points[:, 1] >= 2 * geometry.width_m / 3))]
        team_summary[team] = {
            "frames": len(team_rows),
            "shape_frames": sum(f"{prefix}_length" in row for row in team_rows),
            "position_samples": len(all_points[team]),
            "median_visible": median_or_none([row[f"{prefix}_visible"] for row in team_rows]),
            "mean_centroid_x": mean_or_none([row[f"{prefix}_centroid_x"] for row in team_rows if f"{prefix}_centroid_x" in row]),
            "mean_centroid_y": mean_or_none([row[f"{prefix}_centroid_y"] for row in team_rows if f"{prefix}_centroid_y" in row]),
            "median_length": median_or_none([row[f"{prefix}_length"] for row in team_rows if f"{prefix}_length" in row]),
            "median_width": median_or_none([row[f"{prefix}_width"] for row in team_rows if f"{prefix}_width" in row]),
            "median_compactness": median_or_none([row[f"{prefix}_compactness"] for row in team_rows if f"{prefix}_compactness" in row]),
            "x_thirds": thirds_x,
            "y_lanes": lanes_y,
        }

    duration_seconds = (
        max(
            (float(item.get("timestamp_seconds", 0.0)) for item in calibrations[:frame_count]),
            default=0.0,
        )
        if calibrations
        else frame_count / args.fps
    )

    # Minute-level robust aggregates.
    minute_buckets: dict[int, list[dict]] = defaultdict(list)
    for row in frame_rows:
        minute_buckets[int(row["second"] // 60)].append(row)
    minute_rows = []
    for minute in range(max(1, int(np.ceil(duration_seconds / 60)))):
        rows = minute_buckets.get(minute, [])
        output = {
            "minute": minute,
            "qualified_frames": len(rows),
            "both_teams_frames": sum("centroid_separation" in row for row in rows),
        }
        for team in team_names:
            prefix = team.lower()
            for metric in ("visible", "centroid_x", "centroid_y", "length", "width", "compactness"):
                values = [row[f"{prefix}_{metric}"] for row in rows if f"{prefix}_{metric}" in row]
                output[f"{prefix}_{metric}"] = median_or_none(values)
        output["centroid_separation"] = median_or_none(
            [row["centroid_separation"] for row in rows if "centroid_separation" in row]
        )
        minute_rows.append(output)

    csv_fields = list(minute_rows[0].keys())
    with (args.output_dir / "minute_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(minute_rows)

    # Team heatmaps.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, team in zip(axes, team_names):
        heat = smooth_histogram(all_points[team], geometry)
        draw_pitch(axis, geometry)
        axis.imshow(
            heat,
            extent=(0, geometry.length_m, geometry.width_m, 0),
            cmap="magma",
            alpha=0.72,
            aspect="auto",
        )
        axis.set_title(f"{team} team spatial density ({len(all_points[team]):,} accepted positions)")
    fig.savefig(args.output_dir / "team_heatmaps.png", dpi=180)
    plt.close(fig)

    # Minute-by-minute team centres.
    fig, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    minutes = [row["minute"] for row in minute_rows]
    for team in team_names:
        values = [row[f"{team.lower()}_centroid_x"] for row in minute_rows]
        axis.plot(minutes, values, marker="o", ms=3, lw=1.5, color=colors[team], label=team)
    axis.set_ylim(0, geometry.length_m)
    axis.set_xlabel("Match minute")
    axis.set_ylabel("Median fixed-pitch x centroid (m)")
    axis.set_title("Team longitudinal centres by minute (not attacking direction)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(args.output_dir / "team_centres_timeline.png", dpi=180)
    plt.close(fig)

    # Width and length timeline.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for team in team_names:
        prefix = team.lower()
        axes[0].plot(minutes, [row[f"{prefix}_width"] for row in minute_rows],
                     color=colors[team], label=team)
        axes[1].plot(minutes, [row[f"{prefix}_length"] for row in minute_rows],
                     color=colors[team], label=team)
    axes[0].set_ylabel("Team width (m)")
    axes[0].set_title("Robust team span by minute")
    axes[1].set_ylabel("Team length (m)")
    axes[1].set_xlabel("Match minute")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(args.output_dir / "team_shape_timeline.png", dpi=180)
    plt.close(fig)

    separation_minutes = [
        row for row in minute_rows
        if row["centroid_separation"] is not None and row["both_teams_frames"] >= 5
    ]
    stretched = sorted(separation_minutes, key=lambda row: row["centroid_separation"], reverse=True)[:3]
    compressed = sorted(separation_minutes, key=lambda row: row["centroid_separation"])[:3]

    summary = {
        "source": str(args.source),
        "sample_fps": args.fps,
        "frames": frame_count,
        "duration_seconds": duration_seconds,
        "pitch_length_m": geometry.length_m,
        "pitch_width_m": geometry.width_m,
        "accepted_homography_frames": accepted,
        "accepted_homography_rate": accepted / frame_count if frame_count else 0,
        "median_calibration_quality": median_or_none(accepted_quality),
        "frames_with_both_teams": usable_both,
        "teams": team_summary,
        "most_separated_minutes": stretched,
        "most_compressed_minutes": compressed,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    def minute_list(rows: list[dict]) -> str:
        return "、".join(f"{row['minute']:02d}:00（{row['centroid_separation']:.1f} m）" for row in rows) or "—"

    red = team_summary[team_names[0]]
    blue = team_summary[team_names[1]]
    report = f"""# {args.source.name} 足球视频战术分析

## 结论摘要

- 视频约 {duration_seconds / 60:.1f} 分钟，共分析 {frame_count:,} 个时间样本。
- 其中 {accepted:,} 帧通过场地关键点与单应性质量门槛，占 {(accepted / frame_count * 100) if frame_count else 0.0:.1f}%；{usable_both:,} 帧同时得到两队至少 2 名场内球员的有效重心位置。
- 质量筛选后获得红队 {red['position_samples']:,} 个、蓝队 {blue['position_samples']:,} 个场内位置样本，可用于观察大范围空间偏好；相机跟拍偏差仍然存在。
- 在至少识别到 3 名同队球员的画面中，红队典型阵型跨度约为纵向 {fmt(red['median_length'], ' m')}、横向 {fmt(red['median_width'], ' m')}（{red['shape_frames']} 帧）；蓝队约为纵向 {fmt(blue['median_length'], ' m')}、横向 {fmt(blue['median_width'], ' m')}（{blue['shape_frames']} 帧）。
- 两队纵向重心分离最大的分钟：{minute_list(stretched)}；最压缩的分钟：{minute_list(compressed)}。该指标反映固定场地坐标上的队形拉开/靠拢，不代表控球或攻守优劣。

## 战术解读

1. 固定场地上侧三分之一的有效位置占比：红队 {red['y_lanes'][0]*100:.1f}%，蓝队 {blue['y_lanes'][0]*100:.1f}%。这是场地空间分布，不直接表示攻守方向。
2. 两队纵向重心分离最大的分钟：{minute_list(stretched)}；最压缩的分钟：{minute_list(compressed)}。建议回看这些时段核对阵线脱节、接应和局部压迫。
3. 阵型宽度、长度只在同队至少 3 名球员有有效米制位置时统计；需结合原视频判读，不应单独作为战术结论。

## 团队空间统计

| 指标 | 红队 | 蓝队 |
|---|---:|---:|
| 有效位置样本 | {red['position_samples']:,} | {blue['position_samples']:,} |
| 固定场地 x 重心均值 | {fmt(red['mean_centroid_x'], ' m')} | {fmt(blue['mean_centroid_x'], ' m')} |
| 固定场地 y 重心均值 | {fmt(red['mean_centroid_y'], ' m')} | {fmt(blue['mean_centroid_y'], ' m')} |
| 典型纵向跨度（P90-P10） | {fmt(red['median_length'], ' m')} | {fmt(blue['median_length'], ' m')} |
| 典型横向跨度（P90-P10） | {fmt(red['median_width'], ' m')} | {fmt(blue['median_width'], ' m')} |
| 典型平均紧凑半径 | {fmt(red['median_compactness'], ' m')} | {fmt(blue['median_compactness'], ' m')} |

固定场地左/中/右三区位置占比：红队 {red['x_thirds'][0]*100:.1f}% / {red['x_thirds'][1]*100:.1f}% / {red['x_thirds'][2]*100:.1f}%；蓝队 {blue['x_thirds'][0]*100:.1f}% / {blue['x_thirds'][1]*100:.1f}% / {blue['x_thirds'][2]*100:.1f}%。由于未可靠识别双方逐时段进攻方向，这里只作为空间分布，不命名为进攻三区或防守三区。

## 如何查看产物

- `team_heatmaps.png`：两队通过质量筛选后的位置密度。
- `team_centres_timeline.png`：每分钟纵向重心变化。
- `team_shape_timeline.png`：每分钟横向宽度与纵向长度。
- `minute_metrics.csv`：可进一步筛选的逐分钟数据。
- 标注视频：保留输入尺寸与时间轴，显示球员检测、分队与俯视投影。

## 可信度与限制

1. 这是 XbotGo 自动跟拍的单机位视频；镜头视野随球移动，因此热图仍有“相机关注区域”偏差，不能等同于全场 GPS 数据。
2. 足球由独立高分辨率模型与短时跟踪生成；控球结果仍应在人工校验球检出精度后使用。
3. 边裁及贴近边线的场外人员偶尔会被识别成球员。统计已要求投影落在场地边界内并使用场地关键点质量门槛，但仍可能有少量残留。
4. 低帧率策略用于团队空间趋势，不适合个人跑动距离、冲刺次数和瞬时速度；这些指标已主动关闭。
5. 球员编号是短期跟踪 ID，不是球衣号码，不能据此做个人级结论。
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
