"""Local-only web GUI for the Football Analysis command-line tools."""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge


ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MODEL_EXTENSIONS = {".pt", ".onnx"}
ARTIFACT_EXTENSIONS = {
    ".mp4",
    ".png",
    ".jpg",
    ".jpeg",
    ".json",
    ".csv",
    ".md",
    ".jsonl",
    ".txt",
}
SUMMARY_FILENAMES = (
    "summary.json",
    "minute_metrics.csv",
    "team_heatmaps.png",
    "team_centres_timeline.png",
    "team_shape_timeline.png",
    "REPORT.md",
)
ACTIVE_STATUSES = {"starting", "running"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
MAX_JSON_BYTES = 1024 * 1024
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ApiProblem(Exception):
    """An error that should be returned to an API client as JSON."""

    def __init__(
        self,
        message: str,
        status: int = 400,
        code: str = "invalid_request",
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.details = details


@dataclass
class JobSpec:
    job_type: str
    command: list[str]
    options: dict[str, Any]
    expected_artifacts: list[Path]
    progress_file: Path | None = None
    total_frames: int | None = None


@dataclass
class Job:
    id: str
    spec: JobSpec
    status: str = "starting"
    phase: str = "准备启动"
    progress: float | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    cancel_requested: bool = False
    process: Any = None
    log_sequence: int = 0
    logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=5000))
    progress_bytes: int = 0
    progress_lines: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def type(self) -> str:
        return self.spec.job_type

    def append_log(self, text: str) -> None:
        cleaned = text.rstrip("\r\n")
        if not cleaned:
            return
        with self.lock:
            self.log_sequence += 1
            self.logs.append((self.log_sequence, cleaned))


class Repository:
    """Validated access to the three repository-owned file areas."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.input_root = (self.root / "input_videos").resolve()
        self.model_root = (self.root / "models" / "weights").resolve()
        self.output_root = (self.root / "output_videos").resolve()
        self.html_path = self.root / "gui" / "index.html"
        self.upload_lock = threading.Lock()

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            return os.path.commonpath((str(path), str(root))) == str(root)
        except ValueError:
            return False

    def resolve_path(
        self,
        raw_value: Any,
        allowed_root: Path,
        *,
        must_exist: bool = False,
        file_only: bool = False,
        directory_only: bool = False,
    ) -> Path:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ApiProblem("路径不能为空")
        raw_value = raw_value.strip().replace("\\", "/")
        if any(ord(character) < 32 for character in raw_value):
            raise ApiProblem("路径包含不可用的控制字符", code="unsafe_path")
        candidate = Path(raw_value)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise ApiProblem("只允许使用仓库内的相对路径", code="unsafe_path")
        resolved = (self.root / candidate).resolve(strict=False)
        if not self._inside(resolved, allowed_root):
            raise ApiProblem("路径超出了允许的仓库目录", code="unsafe_path")
        if must_exist and not resolved.exists():
            raise ApiProblem(f"文件或目录不存在：{raw_value}", code="missing_path")
        if file_only and resolved.exists() and not resolved.is_file():
            raise ApiProblem(f"必须选择文件：{raw_value}")
        if directory_only and resolved.exists() and not resolved.is_dir():
            raise ApiProblem(f"必须选择目录：{raw_value}")
        return resolved

    def relative(self, path: Path) -> str:
        return path.resolve(strict=False).relative_to(self.root).as_posix()

    @staticmethod
    def _safe_files(root: Path, extensions: set[str]) -> list[Path]:
        if not root.is_dir():
            return []
        result = []
        for path in root.rglob("*"):
            if ".uploads" in path.relative_to(root).parts:
                continue
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    resolved = path.resolve()
                    if Repository._inside(resolved, root):
                        result.append(resolved)
                except OSError:
                    continue
        return sorted(result, key=lambda item: str(item).lower())

    @staticmethod
    def _video_metadata(path: Path) -> dict[str, float | int | None]:
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return {"fps": None, "frames": None, "duration_seconds": None}
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            duration = frames / fps if fps > 0 else None
            return {
                "fps": round(fps, 3) if fps > 0 else None,
                "frames": frames if frames >= 0 else None,
                "duration_seconds": round(duration, 2) if duration is not None else None,
            }
        except Exception:
            return {"fps": None, "frames": None, "duration_seconds": None}

    def _file_info(self, path: Path, *, include_video_metadata: bool = False) -> dict[str, Any]:
        stat = path.stat()
        info: dict[str, Any] = {
            "name": path.name,
            "path": self.relative(path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }
        if include_video_metadata:
            info.update(self._video_metadata(path))
        return info

    @staticmethod
    def _video_filename(raw_name: Any) -> tuple[str, str]:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ApiProblem("请选择要上传的视频", code="missing_video")

        basename = raw_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        suffix = Path(basename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            formats = "、".join(sorted(extension.lstrip(".").upper() for extension in VIDEO_EXTENSIONS))
            raise ApiProblem(
                f"不支持该视频格式；请选择 {formats}",
                code="unsupported_video_format",
            )

        stem = basename[: -len(suffix)]
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
        stem = stem[:180].rstrip(" .") or "video"
        if stem.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
            stem = f"_{stem}"
        return stem, suffix

    def _unique_video_path(self, stem: str, suffix: str) -> Path:
        candidate = self.input_root / f"{stem}{suffix}"
        index = 1
        while candidate.exists():
            candidate = self.input_root / f"{stem} ({index}){suffix}"
            index += 1
        return candidate

    def store_video(self, uploaded_file: Any) -> dict[str, Any]:
        """Validate and atomically store one uploaded video under input_videos."""
        stem, suffix = self._video_filename(getattr(uploaded_file, "filename", None))
        self.input_root.mkdir(parents=True, exist_ok=True)
        temporary_root = self.input_root / ".uploads"
        temporary_root.mkdir(exist_ok=True)
        temporary_root = temporary_root.resolve()
        if not self._inside(temporary_root, self.input_root):
            raise ApiProblem("上传临时目录不安全", status=500, code="unsafe_upload_directory")
        temporary_path = temporary_root / f"{uuid.uuid4().hex}{suffix}"

        try:
            uploaded_file.save(temporary_path)
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise ApiProblem("上传的视频为空", code="empty_video")

            metadata = self._video_metadata(temporary_path)
            if all(value is None for value in metadata.values()):
                raise ApiProblem(
                    "OpenCV 无法读取该视频，请检查文件是否完整以及编码是否受支持",
                    code="invalid_video",
                )

            with self.upload_lock:
                destination = self._unique_video_path(stem, suffix)
                os.replace(temporary_path, destination)

            info = self._file_info(destination)
            info.update(metadata)
            return info
        except ApiProblem:
            raise
        except OSError as exc:
            raise ApiProblem(
                f"保存视频失败：{exc}", status=500, code="upload_failed"
            ) from exc
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def catalog(self) -> dict[str, Any]:
        videos = [
            self._file_info(path, include_video_metadata=True)
            for path in self._safe_files(self.input_root, VIDEO_EXTENSIONS)
        ]
        field_images = [
            self._file_info(path) for path in self._safe_files(self.input_root, IMAGE_EXTENSIONS)
        ]

        model_paths: set[Path] = set(self._safe_files(self.model_root, MODEL_EXTENSIONS))
        if self.model_root.is_dir():
            for xml_path in self.model_root.rglob("*.xml"):
                try:
                    directory = xml_path.parent.resolve()
                    if self._inside(directory, self.model_root):
                        model_paths.add(directory)
                except OSError:
                    continue
        models = []
        for path in sorted(model_paths, key=lambda item: str(item).lower()):
            if path.is_dir():
                models.append({"name": path.name, "path": self.relative(path), "kind": "OpenVINO"})
            else:
                info = self._file_info(path)
                info["kind"] = path.suffix.lstrip(".").upper()
                models.append(info)

        track_dirs = []
        if self.output_root.is_dir():
            for object_path in self.output_root.rglob("object_tracks.jsonl"):
                directory = object_path.parent.resolve()
                keypoint_path = directory / "keypoint_tracks.jsonl"
                calibration_path = directory / "calibration_tracks.jsonl"
                if self._inside(directory, self.output_root) and keypoint_path.is_file():
                    info = {
                        "name": directory.name,
                        "path": self.relative(directory),
                        "object_size": object_path.stat().st_size,
                        "keypoint_size": keypoint_path.stat().st_size,
                        "has_calibration": calibration_path.is_file(),
                    }
                    if calibration_path.is_file():
                        info["calibration_size"] = calibration_path.stat().st_size
                    track_dirs.append(info)
        track_dirs.sort(key=lambda item: item["path"].lower())

        def first_existing(candidates: tuple[str, ...], allowed_root: Path) -> str | None:
            for candidate in candidates:
                path = (self.root / candidate).resolve(strict=False)
                if path.exists() and self._inside(path, allowed_root):
                    return candidate
            return None

        return {
            "videos": videos,
            "models": models,
            "field_images": field_images,
            "track_dirs": track_dirs,
            "defaults": {
                "object_model": first_existing(
                    (
                        "models/weights/object-detection_openvino_model_fp16",
                        "models/weights/object-detection_openvino_model",
                        "models/weights/object-detection.pt",
                    ),
                    self.model_root,
                ),
                "keypoints_model": first_existing(
                    (
                        "models/weights/keypoints-detection_openvino_model_fp16",
                        "models/weights/keypoints-detection_openvino_model",
                        "models/weights/keypoints-detection.pt",
                    ),
                    self.model_root,
                ),
                "ball_model": first_existing(
                    (
                        "models/weights/ball-detection_openvino_model_fp16",
                        "models/weights/ball-detection_openvino_model",
                        "models/weights/ball-detection.pt",
                    ),
                    self.model_root,
                ),
                "field_image": (
                    "input_videos/field_2d_v2.png"
                    if (self.root / "input_videos/field_2d_v2.png").is_file()
                    else None
                ),
            },
        }

    def results(self) -> dict[str, Any]:
        videos = [
            self._file_info(path, include_video_metadata=True)
            for path in self._safe_files(self.output_root, VIDEO_EXTENSIONS)
        ]
        for item in videos:
            item["url"] = "/artifacts/" + item["path"]

        tracks = []
        if self.output_root.is_dir():
            for object_path in self.output_root.rglob("object_tracks.jsonl"):
                directory = object_path.parent.resolve()
                keypoint_path = directory / "keypoint_tracks.jsonl"
                if not self._inside(directory, self.output_root) or not keypoint_path.is_file():
                    continue
                tracks.append(
                    {
                        "directory": self.relative(directory),
                        "object_tracks": self._file_info(object_path),
                        "keypoint_tracks": self._file_info(keypoint_path),
                    }
                )

        summaries = []
        if self.output_root.is_dir():
            summary_dirs: set[Path] = set()
            for filename in SUMMARY_FILENAMES:
                summary_dirs.update(path.parent.resolve() for path in self.output_root.rglob(filename))
            for directory in sorted(summary_dirs, key=lambda item: str(item).lower()):
                if not self._inside(directory, self.output_root):
                    continue
                files = {}
                for filename in SUMMARY_FILENAMES:
                    path = directory / filename
                    if path.is_file():
                        info = self._file_info(path)
                        info["url"] = "/artifacts/" + info["path"]
                        files[filename] = info
                if files:
                    summaries.append({"directory": self.relative(directory), "files": files})

        videos.sort(key=lambda item: item["modified"], reverse=True)
        tracks.sort(key=lambda item: item["directory"].lower())
        summaries.sort(key=lambda item: item["directory"].lower())
        return {"videos": videos, "tracks": tracks, "summaries": summaries}

    @staticmethod
    def _bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ApiProblem("开关参数必须是布尔值")
        return value

    @staticmethod
    def _integer(value: Any, label: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise ApiProblem(f"{label} 必须是整数")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ApiProblem(f"{label} 必须是整数") from exc
        if result < minimum:
            raise ApiProblem(f"{label} 不能小于 {minimum}")
        return result

    @staticmethod
    def _positive_float(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise ApiProblem(f"{label} 必须是数字")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ApiProblem(f"{label} 必须是数字") from exc
        if not math.isfinite(result) or result <= 0:
            raise ApiProblem(f"{label} 必须是大于 0 的有限数值")
        return result

    @staticmethod
    def _text(value: Any, label: str) -> str:
        if not isinstance(value, str):
            raise ApiProblem(f"{label} 不能为空")
        value = value.strip()
        if not value or len(value) > 100 or any(ord(char) < 32 for char in value):
            raise ApiProblem(f"{label} 必须是 1–100 个可显示字符")
        return value

    @staticmethod
    def _rgb(value: Any, label: str) -> tuple[int, int, int]:
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ApiProblem(f"{label} 必须包含三个 RGB 通道")
        try:
            channels = tuple(int(channel) for channel in value)
        except (TypeError, ValueError) as exc:
            raise ApiProblem(f"{label} 的 RGB 通道必须是整数") from exc
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ApiProblem(f"{label} 的 RGB 通道必须在 0–255 之间")
        return channels

    def _verify_model(self, raw_value: Any, label: str) -> Path:
        path = self.resolve_path(raw_value, self.model_root, must_exist=True)
        if path.is_file() and path.suffix.lower() not in MODEL_EXTENSIONS:
            raise ApiProblem(f"{label} 必须是 .pt/.onnx 文件或 OpenVINO 目录")
        if path.is_dir() and not any(path.glob("*.xml")):
            raise ApiProblem(f"{label} 目录中没有 OpenVINO .xml 模型")
        return path

    def _collision_check(self, paths: list[Path], overwrite: bool) -> None:
        collisions = [self.relative(path) for path in paths if path.exists()]
        if collisions and not overwrite:
            raise ApiProblem(
                "目标文件已经存在，需要确认覆盖",
                status=409,
                code="overwrite_required",
                collisions=collisions,
            )

    def prepare_job(self, payload: Any) -> JobSpec:
        if not isinstance(payload, dict):
            raise ApiProblem("请求内容必须是 JSON 对象")
        job_type = payload.get("type")
        options = payload.get("options", {})
        overwrite = payload.get("overwrite", False)
        if not isinstance(options, dict) or not isinstance(overwrite, bool):
            raise ApiProblem("任务参数格式不正确")

        if job_type == "setup":
            load_models = self._bool(options.get("load_models"), False)
            command = [sys.executable, str(self.root / "check_setup.py")]
            if load_models:
                command.append("--load-models")
            return JobSpec("setup", command, {"load_models": load_models}, [])

        if job_type == "analysis":
            input_path = self.resolve_path(
                options.get("input"), self.input_root, must_exist=True, file_only=True
            )
            if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ApiProblem("输入文件必须是支持的视频格式")
            output_path = self.resolve_path(options.get("output"), self.output_root)
            if output_path.suffix.lower() != ".mp4":
                raise ApiProblem("分析输出必须使用 .mp4 扩展名")
            tracks_dir = self.resolve_path(options.get("tracks_dir"), self.output_root)
            object_model = self._verify_model(options.get("object_model"), "目标检测模型")
            keypoints_model = self._verify_model(options.get("keypoints_model"), "关键点模型")
            ball_model = self._verify_model(options.get("ball_model"), "足球检测模型")
            field_image = self.resolve_path(
                options.get("field_image"), self.input_root, must_exist=True, file_only=True
            )
            if field_image.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ApiProblem("球场底图必须是 PNG 或 JPEG")

            batch_size = self._integer(options.get("batch_size", 1), "批量大小", 1)
            skip_seconds = self._integer(options.get("skip_seconds", 0), "跳过秒数", 0)
            pitch_length_m = self._positive_float(options.get("pitch_length_m"), "球场长度")
            pitch_width_m = self._positive_float(options.get("pitch_width_m"), "球场宽度")
            if pitch_length_m <= 33:
                raise ApiProblem("球场长度必须大于 33 米，以容纳标准禁区")
            if pitch_width_m < 40.32:
                raise ApiProblem("球场宽度不能小于 40.32 米，以容纳标准禁区")
            estimate_speed = self._bool(options.get("estimate_speed"), False)
            annotate_possession = self._bool(options.get("annotate_possession"), False)
            preview = self._bool(options.get("preview"), False)
            club1_name = self._text(options.get("club1_name", "Red"), "第一队名称")
            club2_name = self._text(options.get("club2_name", "Blue"), "第二队名称")
            club1_player = self._rgb(options.get("club1_player", [232, 247, 248]), "第一队球员颜色")
            club1_goalkeeper = self._rgb(
                options.get("club1_goalkeeper", [6, 25, 21]), "第一队门将颜色"
            )
            club2_player = self._rgb(options.get("club2_player", [172, 251, 145]), "第二队球员颜色")
            club2_goalkeeper = self._rgb(
                options.get("club2_goalkeeper", [239, 156, 132]), "第二队门将颜色"
            )

            expected = [
                output_path,
                tracks_dir / "object_tracks.jsonl",
                tracks_dir / "keypoint_tracks.jsonl",
                tracks_dir / "calibration_tracks.jsonl",
            ]
            self._collision_check(expected, overwrite)

            def rel(path: Path) -> str:
                return self.relative(path)

            command = [
                sys.executable,
                str(self.root / "main.py"),
                "--input",
                rel(input_path),
                "--output",
                rel(output_path),
                "--object-model",
                rel(object_model),
                "--keypoints-model",
                rel(keypoints_model),
                "--ball-model",
                rel(ball_model),
                "--field-image",
                rel(field_image),
                "--pitch-length-m",
                str(pitch_length_m),
                "--pitch-width-m",
                str(pitch_width_m),
                "--tracks-dir",
                rel(tracks_dir),
                "--batch-size",
                str(batch_size),
                "--skip-seconds",
                str(skip_seconds),
                "--estimate-speed" if estimate_speed else "--no-estimate-speed",
                "--annotate-possession" if annotate_possession else "--no-annotate-possession",
                "--preview" if preview else "--no-preview",
                "--club1-name",
                club1_name,
                "--club1-player",
                ",".join(map(str, club1_player)),
                "--club1-goalkeeper",
                ",".join(map(str, club1_goalkeeper)),
                "--club2-name",
                club2_name,
                "--club2-player",
                ",".join(map(str, club2_player)),
                "--club2-goalkeeper",
                ",".join(map(str, club2_goalkeeper)),
            ]
            metadata = self._video_metadata(input_path)
            total_frames = metadata.get("frames")
            fps = metadata.get("fps")
            if isinstance(total_frames, int) and isinstance(fps, (int, float)) and fps > 0:
                total_frames = max(0, total_frames - int(skip_seconds * fps))
            else:
                total_frames = None
            normalized_options = {
                "input": rel(input_path),
                "output": rel(output_path),
                "object_model": rel(object_model),
                "keypoints_model": rel(keypoints_model),
                "ball_model": rel(ball_model),
                "field_image": rel(field_image),
                "pitch_length_m": pitch_length_m,
                "pitch_width_m": pitch_width_m,
                "tracks_dir": rel(tracks_dir),
                "batch_size": batch_size,
                "skip_seconds": skip_seconds,
                "estimate_speed": estimate_speed,
                "annotate_possession": annotate_possession,
                "preview": preview,
                "club1_name": club1_name,
                "club1_player": list(club1_player),
                "club1_goalkeeper": list(club1_goalkeeper),
                "club2_name": club2_name,
                "club2_player": list(club2_player),
                "club2_goalkeeper": list(club2_goalkeeper),
                "source_fps": fps,
            }
            return JobSpec(
                "analysis",
                command,
                normalized_options,
                expected,
                progress_file=tracks_dir / "object_tracks.jsonl",
                total_frames=total_frames,
            )

        if job_type == "summary":
            tracks_dir = self.resolve_path(
                options.get("tracks_dir"), self.output_root, must_exist=True, directory_only=True
            )
            for filename in ("object_tracks.jsonl", "keypoint_tracks.jsonl"):
                if not (tracks_dir / filename).is_file():
                    raise ApiProblem(f"轨迹目录缺少 {filename}")
            output_dir = self.resolve_path(options.get("output_dir"), self.output_root)
            source = self.resolve_path(
                options.get("source"), self.input_root, must_exist=True, file_only=True
            )
            if source.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ApiProblem("源文件必须是支持的视频格式")
            fps = self._positive_float(options.get("fps", 1), "采样 FPS")
            pitch_length_m = self._positive_float(options.get("pitch_length_m"), "球场长度")
            pitch_width_m = self._positive_float(options.get("pitch_width_m"), "球场宽度")
            if pitch_length_m <= 33:
                raise ApiProblem("球场长度必须大于 33 米，以容纳标准禁区")
            if pitch_width_m < 40.32:
                raise ApiProblem("球场宽度不能小于 40.32 米，以容纳标准禁区")
            expected = [output_dir / filename for filename in SUMMARY_FILENAMES]
            self._collision_check(expected, overwrite)
            normalized_options = {
                "tracks_dir": self.relative(tracks_dir),
                "output_dir": self.relative(output_dir),
                "source": self.relative(source),
                "fps": fps,
                "pitch_length_m": pitch_length_m,
                "pitch_width_m": pitch_width_m,
            }
            command = [
                sys.executable,
                str(self.root / "summarize_match.py"),
                "--tracks-dir",
                normalized_options["tracks_dir"],
                "--output-dir",
                normalized_options["output_dir"],
                "--fps",
                str(fps),
                "--source",
                normalized_options["source"],
                "--pitch-length-m",
                str(pitch_length_m),
                "--pitch-width-m",
                str(pitch_width_m),
            ]
            return JobSpec("summary", command, normalized_options, expected)

        raise ApiProblem("任务类型必须是 setup、analysis 或 summary")


class JobManager:
    """Run one supervised CLI subprocess at a time."""

    def __init__(
        self,
        repository: Repository,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.repository = repository
        self.popen_factory = popen_factory
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.active_job_id: str | None = None
        self.lock = threading.RLock()

    def create(self, spec: JobSpec) -> Job:
        with self.lock:
            if self.active_job_id:
                active = self.jobs.get(self.active_job_id)
                if active and active.status in ACTIVE_STATUSES:
                    raise ApiProblem(
                        "已有任务正在运行",
                        status=409,
                        code="active_job",
                        active_job_id=active.id,
                    )
            job = Job(id=uuid.uuid4().hex[:12], spec=spec)
            self.jobs[job.id] = job
            self.active_job_id = job.id
            while len(self.jobs) > 20:
                oldest_id, oldest = next(iter(self.jobs.items()))
                if oldest.status in ACTIVE_STATUSES:
                    break
                self.jobs.pop(oldest_id)
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
            return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise ApiProblem("找不到该任务", status=404, code="job_not_found")
        return job

    def list(self) -> list[Job]:
        with self.lock:
            return list(reversed(self.jobs.values()))

    @staticmethod
    def _set_phase_from_log(job: Job, line: str) -> None:
        lowered = line.lower()
        with job.lock:
            if job.type == "analysis":
                if "starting frame capture" in lowered:
                    job.phase = "读取视频"
                elif "starting frame processing" in lowered:
                    job.phase = "分析视频"
                elif "converting frames to video" in lowered:
                    job.phase = "封装视频"
                    job.progress = None
                elif "video saved as" in lowered:
                    job.phase = "核对产物"
            elif job.type == "setup":
                job.phase = "检查环境"
            elif job.type == "summary":
                job.phase = "生成战术总结"

    def _run(self, job: Job) -> None:
        with job.lock:
            if job.cancel_requested:
                job.status = "cancelled"
                job.phase = "已取消"
                job.finished_at = time.time()
                self._release(job)
                return
            job.status = "running"
            job.started_at = time.time()
            job.phase = {
                "setup": "检查环境",
                "analysis": "加载模型",
                "summary": "生成战术总结",
            }[job.type]

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["MPLBACKEND"] = "Agg"
        kwargs: dict[str, Any] = {
            "cwd": str(self.repository.root),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": environment,
            "shell": False,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        try:
            process = self.popen_factory(job.spec.command, **kwargs)
            with job.lock:
                job.process = process
            if job.cancel_requested:
                self._signal_stop(job)
            if process.stdout is not None:
                for line in process.stdout:
                    job.append_log(line)
                    self._set_phase_from_log(job, line)
            return_code = process.wait()
            with job.lock:
                job.return_code = return_code
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.phase = "已取消（可能保留部分产物）"
                elif return_code != 0:
                    job.status = "failed"
                    job.phase = "运行失败"
                else:
                    fatal_markers = (
                        "error: could not open video source",
                        "error processing batch:",
                        "error in frame capture:",
                        "error in frame processing:",
                        "error in frame display:",
                        "an error occurred:",
                        "traceback (most recent call last)",
                    )
                    logged_failure = job.type == "analysis" and any(
                        any(marker in text.lower() for marker in fatal_markers)
                        for _sequence, text in job.logs
                    )
                    missing = [
                        path
                        for path in job.spec.expected_artifacts
                        if not path.is_file() or path.stat().st_size <= 0
                    ]
                    if logged_failure:
                        job.status = "failed"
                        job.phase = "分析过程中出现错误"
                    elif missing:
                        job.status = "failed"
                        job.phase = "产物不完整"
                        for path in missing:
                            job.append_log(f"缺少或为空的预期产物：{self.repository.relative(path)}")
                    else:
                        job.status = "succeeded"
                        job.phase = "已完成"
                        if job.type == "analysis":
                            job.progress = 100.0
        except Exception as exc:
            job.append_log(f"无法启动或监督任务：{exc}")
            with job.lock:
                job.status = "cancelled" if job.cancel_requested else "failed"
                job.phase = "已取消" if job.cancel_requested else "启动失败"
        finally:
            with job.lock:
                job.finished_at = time.time()
            self._release(job)

    def _release(self, job: Job) -> None:
        with self.lock:
            if self.active_job_id == job.id:
                self.active_job_id = None

    def refresh_progress(self, job: Job) -> None:
        if job.type != "analysis" or job.status not in ACTIVE_STATUSES:
            return
        progress_file = job.spec.progress_file
        total = job.spec.total_frames
        if progress_file is None or not total or total <= 0 or not progress_file.is_file():
            return
        try:
            stat = progress_file.stat()
            if job.started_at is not None and stat.st_mtime + 0.001 < job.started_at:
                return
            with job.lock:
                if stat.st_size < job.progress_bytes:
                    job.progress_bytes = 0
                    job.progress_lines = 0
                start = job.progress_bytes
            new_lines = 0
            with progress_file.open("rb") as handle:
                handle.seek(start)
                while chunk := handle.read(1024 * 1024):
                    new_lines += chunk.count(b"\n")
                end = handle.tell()
            with job.lock:
                job.progress_bytes = end
                job.progress_lines += new_lines
                job.progress = min(99.0, round(job.progress_lines / total * 100, 1))
                if job.progress_lines > 0 and job.phase in {"加载模型", "读取视频"}:
                    job.phase = "分析视频"
        except OSError:
            return

    def _signal_stop(self, job: Job) -> None:
        process = job.process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except Exception:
            try:
                process.terminate()
            except Exception:
                return

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with job.lock:
            if job.status not in ACTIVE_STATUSES:
                raise ApiProblem("任务已经结束", status=409, code="job_finished")
            job.cancel_requested = True
            job.phase = "正在停止"
        self._signal_stop(job)

        def force_kill() -> None:
            time.sleep(5)
            process = job.process
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    job.append_log("任务未在 5 秒内停止，已强制结束。")
                except Exception:
                    pass

        threading.Thread(target=force_kill, daemon=True).start()
        return job

    def snapshot(self, job: Job, after: int = 0, include_logs: bool = True) -> dict[str, Any]:
        self.refresh_progress(job)
        with job.lock:
            logs = []
            if include_logs:
                logs = [{"seq": seq, "text": text} for seq, text in job.logs if seq > after]
            artifacts = []
            for path in job.spec.expected_artifacts:
                if path.is_file():
                    relative = self.repository.relative(path)
                    artifacts.append(
                        {
                            "path": relative,
                            "url": "/artifacts/" + relative,
                            "size": path.stat().st_size,
                        }
                    )
            return {
                "id": job.id,
                "type": job.type,
                "status": job.status,
                "phase": job.phase,
                "progress": job.progress,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "return_code": job.return_code,
                "options": job.spec.options,
                "artifacts": artifacts,
                "logs": logs,
                "next_log_seq": job.log_sequence,
            }


def create_app(
    root: Path = ROOT,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> Flask:
    repository = Repository(root)
    manager = JobManager(repository, popen_factory)
    app = Flask(__name__)
    app.config.update(JSON_AS_ASCII=False, MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES)
    app.extensions["football_repository"] = repository
    app.extensions["football_jobs"] = manager

    @app.errorhandler(ApiProblem)
    def handle_api_problem(problem: ApiProblem) -> tuple[Response, int]:
        payload = {"error": problem.message, "code": problem.code, **problem.details}
        return jsonify(payload), problem.status

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_problem: RequestEntityTooLarge) -> tuple[Response, int]:
        limit = int(app.config["MAX_CONTENT_LENGTH"])
        limit_label = f"{limit // (1024 ** 3)} GiB" if limit >= 1024 ** 3 else f"{limit} 字节"
        return (
            jsonify(
                {
                    "error": f"视频超过 {limit_label} 上传上限",
                    "code": "upload_too_large",
                    "max_bytes": limit,
                }
            ),
            413,
        )

    @app.before_request
    def protect_local_writes() -> tuple[Response, int] | None:
        hostname = request.host.split(":", 1)[0].strip("[]").lower()
        if hostname not in {"127.0.0.1", "localhost"}:
            return jsonify({"error": "只允许通过本机地址访问", "code": "host_rejected"}), 403
        if request.method not in {"POST", "DELETE"}:
            return None
        origin = request.headers.get("Origin")
        expected = f"{request.scheme}://{request.host}"
        if origin and origin.rstrip("/") != expected.rstrip("/"):
            return jsonify({"error": "拒绝跨来源请求", "code": "origin_rejected"}), 403
        if request.method == "POST":
            if request.path == "/api/videos":
                if request.mimetype != "multipart/form-data":
                    return jsonify(
                        {"error": "视频上传必须使用 multipart/form-data", "code": "multipart_required"}
                    ), 415
            else:
                if request.content_length is not None and request.content_length > MAX_JSON_BYTES:
                    return jsonify({"error": "JSON 请求过大", "code": "request_too_large"}), 413
                if not request.is_json:
                    return jsonify({"error": "请求必须使用 JSON", "code": "json_required"}), 415
        return None

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> Response:
        if not repository.html_path.is_file():
            raise ApiProblem("GUI 页面文件不存在", status=500, code="missing_gui")
        return send_file(repository.html_path)

    @app.get("/api/catalog")
    def catalog() -> Response:
        return jsonify(repository.catalog())

    @app.get("/api/results")
    def results() -> Response:
        return jsonify(repository.results())

    @app.post("/api/videos")
    def upload_video() -> tuple[Response, int]:
        uploaded_file = request.files.get("video")
        if uploaded_file is None:
            raise ApiProblem("请求中缺少 video 文件字段", code="missing_video")
        return jsonify({"video": repository.store_video(uploaded_file)}), 201

    @app.get("/api/jobs")
    def jobs() -> Response:
        return jsonify({"jobs": [manager.snapshot(job, include_logs=False) for job in manager.list()]})

    @app.post("/api/jobs")
    def create_job() -> tuple[Response, int]:
        spec = repository.prepare_job(request.get_json(silent=False))
        job = manager.create(spec)
        return jsonify(manager.snapshot(job)), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str) -> Response:
        try:
            after = max(0, int(request.args.get("after", "0")))
        except ValueError as exc:
            raise ApiProblem("after 必须是非负整数") from exc
        return jsonify(manager.snapshot(manager.get(job_id), after=after))

    @app.delete("/api/jobs/<job_id>")
    def cancel_job(job_id: str) -> Response:
        return jsonify(manager.snapshot(manager.cancel(job_id)))

    @app.get("/artifacts/<path:artifact_path>")
    def artifact(artifact_path: str) -> Response:
        path = repository.resolve_path(
            artifact_path, repository.output_root, must_exist=True, file_only=True
        )
        if path.suffix.lower() not in ARTIFACT_EXTENSIONS:
            raise ApiProblem("不允许访问该文件类型", status=403, code="artifact_forbidden")
        mimetype = None
        if path.suffix.lower() == ".md":
            mimetype = "text/markdown; charset=utf-8"
        return send_file(path, conditional=True, mimetype=mimetype, download_name=path.name)

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Football Analysis web GUI.")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the GUI in the default browser"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Football Analysis GUI: {url}")
    print("Press Ctrl+C to stop the local server.")
    create_app().run(
        host="127.0.0.1",
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
