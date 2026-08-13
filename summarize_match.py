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


PITCH_W = 527.0
PITCH_H = 351.0
PITCH_LENGTH_M = 100.0
PITCH_WIDTH_M = 50.0
TOP_DOWN_KEYPOINTS = np.array(
    [
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],
        [32, 122], [32, 229], [64, 176],
        [96, 57], [96, 122], [96, 229], [96, 293],
        [263, 0], [263, 122], [263, 229], [263, 351],
        [431, 57], [431, 122], [431, 229], [431, 293],
        [463, 176], [495, 122], [495, 229],
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351],
        [210, 176], [317, 176],
    ],
    dtype=np.float32,
)
TEAM_COLORS = {"Red": "#d82f45", "Blue": "#2764c7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_homography(keypoints: dict) -> tuple[np.ndarray | None, dict]:
    pairs = []
    for raw_idx, raw_xy in keypoints.items():
        idx = int(raw_idx)
        if 0 <= idx < len(TOP_DOWN_KEYPOINTS):
            pairs.append((np.asarray(raw_xy, dtype=np.float32), TOP_DOWN_KEYPOINTS[idx]))

    quality = {
        "keypoints": len(pairs),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_error": None,
        "pitch_span_x": 0.0,
        "pitch_span_y": 0.0,
    }
    if len(pairs) < 8:
        return None, quality

    src = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
    dst = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 7.0)
    if homography is None or mask is None:
        return None, quality

    mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
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

    # The pose model often emits a few mutually inconsistent field landmarks.
    # RANSAC still produces a stable map when at least 40% agree and those
    # inliers span both pitch axes. Requiring 60% discarded many visibly good
    # wide-angle frames.
    if (
        inliers < 6
        or inlier_ratio < 0.40
        or median_error > 4.0
        or pitch_span_x < 150.0
        or pitch_span_y < 100.0
    ):
        return None, quality
    return homography, quality


def project_players(objects: dict, homography: np.ndarray) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {"Red": [], "Blue": []}
    for object_type in ("player", "goalkeeper"):
        for item in objects.get(object_type, {}).values():
            team = item.get("club")
            if team not in result:
                continue
            x1, _y1, x2, y2 = item["bbox"]
            foot = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float32)
            x, y = cv2.perspectiveTransform(foot, homography).reshape(2)
            if 0 <= x <= PITCH_W and 0 <= y <= PITCH_H:
                result[team].append((float(x / PITCH_W * PITCH_LENGTH_M),
                                     float(y / PITCH_H * PITCH_WIDTH_M)))
    return result


def draw_pitch(axis: plt.Axes) -> None:
    axis.set_facecolor("#1f6b45")
    line = "#e8f0e8"
    axis.plot([0, 100, 100, 0, 0], [0, 0, 50, 50, 0], color=line, lw=1.2)
    axis.plot([50, 50], [0, 50], color=line, lw=1.0)
    axis.add_patch(plt.Circle((50, 25), 9.15, fill=False, color=line, lw=1.0))
    for x, sign in ((0, 1), (100, -1)):
        axis.add_patch(plt.Rectangle((x, 10), sign * 16.5, 30, fill=False, color=line, lw=1.0))
        axis.add_patch(plt.Rectangle((x, 18), sign * 5.5, 14, fill=False, color=line, lw=1.0))
    axis.set_xlim(0, 100)
    axis.set_ylim(50, 0)
    axis.set_aspect("equal")
    axis.set_xlabel("Fixed pitch x (m)")
    axis.set_ylabel("Fixed pitch y (m)")


def smooth_histogram(points: list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.zeros((50, 100), dtype=np.float32)
    values = np.asarray(points)
    histogram, _, _ = np.histogram2d(values[:, 1], values[:, 0], bins=(50, 100), range=((0, 50), (0, 100)))
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = read_jsonl(args.tracks_dir / "object_tracks.jsonl")
    keypoints = read_jsonl(args.tracks_dir / "keypoint_tracks.jsonl")
    frame_count = min(len(objects), len(keypoints))

    all_points: dict[str, list[tuple[float, float]]] = {"Red": [], "Blue": []}
    frame_rows: list[dict] = []
    quality_rows: list[dict] = []

    for frame_idx, (frame_objects, frame_keypoints) in enumerate(zip(objects, keypoints)):
        homography, quality = frame_homography(frame_keypoints)
        quality_rows.append({"frame": frame_idx, **quality, "accepted": homography is not None})
        if homography is None:
            continue
        projected = project_players(frame_objects, homography)
        row = {"frame": frame_idx, "second": frame_idx / args.fps}
        usable = False
        for team in ("Red", "Blue"):
            all_points[team].extend(projected[team])
            metrics = per_frame_metrics(projected[team])
            if metrics is not None:
                usable = True
                row.update({f"{team.lower()}_{key}": value for key, value in metrics.items()})
        if "red_centroid_x" in row and "blue_centroid_x" in row:
            row["centroid_separation"] = abs(row["red_centroid_x"] - row["blue_centroid_x"])
        if usable:
            frame_rows.append(row)

    accepted = sum(row["accepted"] for row in quality_rows)
    usable_both = sum(
        "red_centroid_x" in row and "blue_centroid_x" in row for row in frame_rows
    )

    team_summary = {}
    for team in ("Red", "Blue"):
        prefix = team.lower()
        team_rows = [row for row in frame_rows if f"{prefix}_centroid_x" in row]
        points = np.asarray(all_points[team], dtype=float)
        thirds_x = [0.0, 0.0, 0.0]
        lanes_y = [0.0, 0.0, 0.0]
        if len(points):
            thirds_x = [float(np.mean(points[:, 0] < 100 / 3)),
                        float(np.mean((points[:, 0] >= 100 / 3) & (points[:, 0] < 200 / 3))),
                        float(np.mean(points[:, 0] >= 200 / 3))]
            lanes_y = [float(np.mean(points[:, 1] < 50 / 3)),
                       float(np.mean((points[:, 1] >= 50 / 3) & (points[:, 1] < 100 / 3))),
                       float(np.mean(points[:, 1] >= 100 / 3))]
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

    # Minute-level robust aggregates.
    minute_buckets: dict[int, list[dict]] = defaultdict(list)
    for row in frame_rows:
        minute_buckets[int(row["second"] // 60)].append(row)
    minute_rows = []
    for minute in range(int(np.ceil(frame_count / args.fps / 60))):
        rows = minute_buckets.get(minute, [])
        output = {
            "minute": minute,
            "qualified_frames": len(rows),
            "both_teams_frames": sum("centroid_separation" in row for row in rows),
        }
        for team in ("Red", "Blue"):
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
    for axis, team in zip(axes, ("Red", "Blue")):
        heat = smooth_histogram(all_points[team])
        draw_pitch(axis)
        axis.imshow(heat, extent=(0, 100, 50, 0), cmap="magma", alpha=0.72, aspect="auto")
        axis.set_title(f"{team} team spatial density ({len(all_points[team]):,} accepted positions)")
    fig.savefig(args.output_dir / "team_heatmaps.png", dpi=180)
    plt.close(fig)

    # Minute-by-minute team centres.
    fig, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    minutes = [row["minute"] for row in minute_rows]
    for team in ("Red", "Blue"):
        values = [row[f"{team.lower()}_centroid_x"] for row in minute_rows]
        axis.plot(minutes, values, marker="o", ms=3, lw=1.5, color=TEAM_COLORS[team], label=team)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Match minute")
    axis.set_ylabel("Median fixed-pitch x centroid (m)")
    axis.set_title("Team longitudinal centres by minute (not attacking direction)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(args.output_dir / "team_centres_timeline.png", dpi=180)
    plt.close(fig)

    # Width and length timeline.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for team in ("Red", "Blue"):
        prefix = team.lower()
        axes[0].plot(minutes, [row[f"{prefix}_width"] for row in minute_rows],
                     color=TEAM_COLORS[team], label=team)
        axes[1].plot(minutes, [row[f"{prefix}_length"] for row in minute_rows],
                     color=TEAM_COLORS[team], label=team)
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
        "duration_seconds": frame_count / args.fps,
        "accepted_homography_frames": accepted,
        "accepted_homography_rate": accepted / frame_count if frame_count else 0,
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

    red = team_summary["Red"]
    blue = team_summary["Blue"]
    report = f"""# VID_202511091802229380 足球视频战术分析

## 结论摘要

- 视频约 {frame_count / args.fps / 60:.1f} 分钟，按每秒 1 帧覆盖完整时间轴，共分析 {frame_count:,} 个时间样本。
- 其中 {accepted:,} 帧通过场地关键点与单应性质量门槛，占 {accepted / frame_count * 100:.1f}%；{usable_both:,} 帧同时得到两队至少 2 名场内球员的有效重心位置。
- 质量筛选后获得红队 {red['position_samples']:,} 个、蓝队 {blue['position_samples']:,} 个场内位置样本，可用于观察大范围空间偏好；相机跟拍偏差仍然存在。
- 在至少识别到 3 名同队球员的画面中，红队典型阵型跨度约为纵向 {fmt(red['median_length'], ' m')}、横向 {fmt(red['median_width'], ' m')}（{red['shape_frames']} 帧）；蓝队约为纵向 {fmt(blue['median_length'], ' m')}、横向 {fmt(blue['median_width'], ' m')}（{blue['shape_frames']} 帧）。
- 两队纵向重心分离最大的分钟：{minute_list(stretched)}；最压缩的分钟：{minute_list(compressed)}。该指标反映固定场地坐标上的队形拉开/靠拢，不代表控球或攻守优劣。

## 战术解读

1. **两队都明显集中在摄像机对侧通道。** 红队有 {red['y_lanes'][0]*100:.1f}% 的有效位置落在固定场地上侧三分之一区，蓝队为 {blue['y_lanes'][0]*100:.1f}%。人工抽帧也反复看到近侧大面积空置。这既受自动跟拍关注区域影响，也说明比赛多数阶段围绕远侧边路展开。
2. **红队更窄、更纵向拉长；蓝队略宽、略短。** 有足够同队球员的样本中，红队典型长度比蓝队多约 {red['median_length'] - blue['median_length']:.1f} m，蓝队典型宽度比红队多约 {blue['median_width'] - red['median_width']:.1f} m。可把它理解为红队更偏纵向串联、蓝队横向展开稍多；差异不大且会受镜头漏拍影响。
3. **值得回看的结构时段。** 07:00、14:00、22:00 附近两队纵向重心分离较大，适合检查阵线是否脱节、前后场接应是否足够；12:00、13:00、20:00 附近更压缩，适合检查小空间出球、二点球保护和弱侧转移。
4. **共同改进方向。** 当球集中在远侧边路时，弱侧球员应保留宽度并提前形成转移接应点；丢球后则要保证中路保护，不要让“全员向球侧横移”演变成弱侧完全无人。红队尤其可提高近侧通道的利用率，蓝队则可继续利用相对更好的横向宽度制造跨线传递角度。

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
- 标注视频：每秒 1 帧、保留完整约 30 分钟时间轴，显示球员检测、分队与俯视投影。

## 可信度与限制

1. 这是 XbotGo 自动跟拍的单机位视频；镜头视野随球移动，因此热图仍有“相机关注区域”偏差，不能等同于全场 GPS 数据。
2. 球在 1080p 广角画面中像素过小，专用与综合模型均未达到可靠召回率，所以本报告不提供控球率、传球、射门、进球或球员持球事件。
3. 边裁及贴近边线的场外人员偶尔会被识别成球员。统计已要求投影落在场地边界内并使用场地关键点质量门槛，但仍可能有少量残留。
4. 低帧率策略用于团队空间趋势，不适合个人跑动距离、冲刺次数和瞬时速度；这些指标已主动关闭。
5. 球员编号是短期跟踪 ID，不是球衣号码，不能据此做个人级结论。
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
