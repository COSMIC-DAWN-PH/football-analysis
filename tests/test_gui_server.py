import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from gui_server import ApiProblem, Repository, create_app
from position_mappers import PitchGeometry


class ImmediateProcess:
    def __init__(self, command, **_kwargs):
        self.command = command
        self.stdout = io.StringIO("Python: 3.11\nOK: input_videos/field_2d_v2.png\nSetup is ready.\n")
        self.returncode = None

    def wait(self):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def send_signal(self, _signal):
        self.returncode = 130

    def terminate(self):
        self.returncode = 143

    def kill(self):
        self.returncode = 137


class BlockingStream:
    def __init__(self, stopped):
        self.stopped = stopped

    def __iter__(self):
        return self

    def __next__(self):
        self.stopped.wait()
        raise StopIteration


class BlockingProcess:
    def __init__(self, _command, **_kwargs):
        self.stopped = threading.Event()
        self.stdout = BlockingStream(self.stopped)
        self.returncode = None

    def wait(self):
        self.stopped.wait()
        return self.returncode if self.returncode is not None else 130

    def poll(self):
        return self.returncode

    def send_signal(self, _signal):
        self.returncode = 130
        self.stopped.set()

    def terminate(self):
        self.returncode = 143
        self.stopped.set()

    def kill(self):
        self.returncode = 137
        self.stopped.set()


class GuiTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for directory in ("input_videos", "models/weights", "output_videos", "gui"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "input_videos" / "match.mp4").write_bytes(b"fake video")
        (self.root / "input_videos" / "field_2d_v2.png").write_bytes(b"fake image")
        (self.root / "models" / "weights" / "object-detection.pt").write_bytes(b"object")
        (self.root / "models" / "weights" / "keypoints-detection.pt").write_bytes(b"pose")
        (self.root / "models" / "weights" / "ball-detection.pt").write_bytes(b"ball")
        (self.root / "gui" / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
        for script in ("main.py", "check_setup.py", "summarize_match.py"):
            (self.root / script).write_text("", encoding="utf-8")
        self.metadata_patch = mock.patch.object(
            Repository,
            "_video_metadata",
            return_value={"fps": 25.0, "frames": 250, "duration_seconds": 10.0},
        )
        self.metadata_patch.start()
        self.repository = Repository(self.root)

    def tearDown(self):
        self.metadata_patch.stop()
        self.temporary.cleanup()

    def analysis_payload(self, overwrite=False):
        return {
            "type": "analysis",
            "overwrite": overwrite,
            "options": {
                "input": "input_videos/match.mp4",
                "output": "output_videos/demo/demo-analysis.mp4",
                "tracks_dir": "output_videos/demo/tracks",
                "object_model": "models/weights/object-detection.pt",
                "keypoints_model": "models/weights/keypoints-detection.pt",
                "ball_model": "models/weights/ball-detection.pt",
                "field_image": "input_videos/field_2d_v2.png",
                "pitch_length_m": 105,
                "pitch_width_m": 68,
                "batch_size": 2,
                "skip_seconds": 3,
                "estimate_speed": True,
                "annotate_possession": False,
                "preview": False,
                "club1_name": "Red",
                "club1_player": [220, 30, 30],
                "club1_goalkeeper": [20, 20, 20],
                "club2_name": "Blue",
                "club2_player": [30, 80, 220],
                "club2_goalkeeper": [240, 220, 30],
            },
        }


class RepositoryTests(GuiTestCase):
    def test_analysis_command_maps_all_cli_arguments_without_shell(self):
        spec = self.repository.prepare_job(self.analysis_payload())
        self.assertEqual(spec.job_type, "analysis")
        self.assertIn("--estimate-speed", spec.command)
        self.assertIn("--no-annotate-possession", spec.command)
        self.assertIn("--no-preview", spec.command)
        self.assertEqual(
            spec.command[spec.command.index("--ball-model") + 1],
            "models/weights/ball-detection.pt",
        )
        self.assertEqual(spec.command[spec.command.index("--pitch-length-m") + 1], "105.0")
        self.assertEqual(spec.command[spec.command.index("--pitch-width-m") + 1], "68.0")
        self.assertEqual(spec.command[spec.command.index("--batch-size") + 1], "2")
        self.assertEqual(spec.command[spec.command.index("--club1-player") + 1], "220,30,30")
        self.assertEqual(spec.options["tracks_dir"], "output_videos/demo/tracks")
        self.assertIn(
            self.root / "output_videos" / "demo" / "tracks" / "calibration_tracks.jsonl",
            spec.expected_artifacts,
        )

    def test_analysis_rejects_invalid_metric_pitch_dimensions(self):
        payload = self.analysis_payload()
        payload["options"]["pitch_width_m"] = 40
        with self.assertRaises(ApiProblem) as problem:
            self.repository.prepare_job(payload)
        self.assertIn("40.32", problem.exception.message)

        payload["options"]["pitch_width_m"] = 68
        payload["options"]["pitch_length_m"] = float("nan")
        with self.assertRaises(ApiProblem):
            self.repository.prepare_job(payload)

        payload["options"]["pitch_length_m"] = True
        with self.assertRaises(ApiProblem):
            self.repository.prepare_job(payload)

    def test_paths_cannot_escape_their_owned_root(self):
        with self.assertRaises(ApiProblem) as problem:
            self.repository.resolve_path("input_videos/../main.py", self.repository.input_root)
        self.assertEqual(problem.exception.code, "unsafe_path")

        with self.assertRaises(ApiProblem):
            self.repository.resolve_path(str(self.root / "input_videos" / "match.mp4"), self.repository.input_root)

    def test_existing_targets_require_explicit_overwrite(self):
        target = self.root / "output_videos" / "demo" / "demo-analysis.mp4"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"old")
        with self.assertRaises(ApiProblem) as problem:
            self.repository.prepare_job(self.analysis_payload())
        self.assertEqual(problem.exception.status, 409)
        self.assertEqual(problem.exception.code, "overwrite_required")

        spec = self.repository.prepare_job(self.analysis_payload(overwrite=True))
        self.assertEqual(spec.job_type, "analysis")

    def test_summary_validates_track_pair_and_positive_fps(self):
        tracks = self.root / "output_videos" / "demo" / "tracks"
        tracks.mkdir(parents=True)
        (tracks / "object_tracks.jsonl").write_text("{}\n", encoding="utf-8")
        (tracks / "keypoint_tracks.jsonl").write_text("{}\n", encoding="utf-8")
        payload = {
            "type": "summary",
            "options": {
                "tracks_dir": "output_videos/demo/tracks",
                "output_dir": "output_videos/demo/summary",
                "source": "input_videos/match.mp4",
                "fps": 25,
                "pitch_length_m": 105,
                "pitch_width_m": 68,
            },
        }
        spec = self.repository.prepare_job(payload)
        self.assertEqual(spec.command[spec.command.index("--fps") + 1], "25.0")
        self.assertEqual(spec.command[spec.command.index("--pitch-length-m") + 1], "105.0")
        payload["options"]["fps"] = 0
        with self.assertRaises(ApiProblem):
            self.repository.prepare_job(payload)


class ApiTests(GuiTestCase):
    def test_catalog_results_and_range_artifact(self):
        output = self.root / "output_videos" / "clip.mp4"
        output.write_bytes(b"0123456789")
        app = create_app(self.root, ImmediateProcess)
        app.testing = True
        client = app.test_client()

        catalog = client.get("/api/catalog")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(len(catalog.get_json()["videos"]), 1)
        self.assertEqual(len(catalog.get_json()["models"]), 3)
        self.assertEqual(
            catalog.get_json()["defaults"]["ball_model"],
            "models/weights/ball-detection.pt",
        )

        results = client.get("/api/results").get_json()
        self.assertEqual(results["videos"][0]["path"], "output_videos/clip.mp4")

        response = client.get(
            "/artifacts/output_videos/clip.mp4", headers={"Range": "bytes=2-5"}
        )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b"2345")
        response.close()
        self.assertEqual(client.get("/artifacts/input_videos/match.mp4").status_code, 400)

    def test_setup_job_exposes_incremental_logs_and_success(self):
        app = create_app(self.root, ImmediateProcess)
        app.testing = True
        client = app.test_client()
        created = client.post(
            "/api/jobs", json={"type": "setup", "options": {"load_models": True}}
        )
        self.assertEqual(created.status_code, 202)
        job_id = created.get_json()["id"]

        deadline = time.time() + 2
        job = None
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}?after=0").get_json()
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded")
        self.assertTrue(any("Setup is ready" in item["text"] for item in job["logs"]))
        self.assertGreater(job["next_log_seq"], 0)

        no_new_logs = client.get(
            f"/api/jobs/{job_id}?after={job['next_log_seq']}"
        ).get_json()
        self.assertEqual(no_new_logs["logs"], [])

    def test_only_one_job_runs_and_it_can_be_cancelled(self):
        app = create_app(self.root, BlockingProcess)
        app.testing = True
        client = app.test_client()
        first = client.post("/api/jobs", json={"type": "setup", "options": {}})
        self.assertEqual(first.status_code, 202)
        job_id = first.get_json()["id"]

        second = client.post("/api/jobs", json={"type": "setup", "options": {}})
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.get_json()["code"], "active_job")

        cancelled = client.delete(f"/api/jobs/{job_id}")
        self.assertEqual(cancelled.status_code, 200)
        deadline = time.time() + 2
        status = None
        while time.time() < deadline:
            status = client.get(f"/api/jobs/{job_id}").get_json()["status"]
            if status == "cancelled":
                break
            time.sleep(0.01)
        self.assertEqual(status, "cancelled")

    def test_cross_origin_write_is_rejected(self):
        app = create_app(self.root, ImmediateProcess)
        app.testing = True
        response = app.test_client().post(
            "/api/jobs",
            json={"type": "setup", "options": {}},
            headers={"Origin": "https://example.test"},
        )
        self.assertEqual(response.status_code, 403)

        response = app.test_client().get("/api/catalog", headers={"Host": "attacker.test"})
        self.assertEqual(response.status_code, 403)

        response = app.test_client().post(
            "/api/videos",
            data={"video": (io.BytesIO(b"video"), "match.mp4")},
            content_type="multipart/form-data",
            headers={"Origin": "https://example.test"},
        )
        self.assertEqual(response.status_code, 403)

    def test_video_upload_preserves_unicode_and_renames_collisions(self):
        app = create_app(self.root, ImmediateProcess)
        app.testing = True
        client = app.test_client()

        first = client.post(
            "/api/videos",
            data={"video": (io.BytesIO(b"first video"), "比赛.mp4")},
            content_type="multipart/form-data",
        )
        second = client.post(
            "/api/videos",
            data={"video": (io.BytesIO(b"second video"), "比赛.mp4")},
            content_type="multipart/form-data",
        )
        traversal = client.post(
            "/api/videos",
            data={"video": (io.BytesIO(b"safe video"), "../outside.mp4")},
            content_type="multipart/form-data",
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(traversal.status_code, 201)
        self.assertEqual(first.get_json()["video"]["path"], "input_videos/比赛.mp4")
        self.assertEqual(second.get_json()["video"]["path"], "input_videos/比赛 (1).mp4")
        self.assertEqual(traversal.get_json()["video"]["path"], "input_videos/outside.mp4")
        self.assertEqual((self.root / "input_videos" / "比赛.mp4").read_bytes(), b"first video")
        self.assertEqual(
            (self.root / "input_videos" / "比赛 (1).mp4").read_bytes(), b"second video"
        )
        self.assertFalse((self.root / "outside.mp4").exists())
        self.assertEqual(len(client.get("/api/catalog").get_json()["videos"]), 4)

    def test_video_upload_rejects_invalid_files_and_cleans_temporary_data(self):
        app = create_app(self.root, ImmediateProcess)
        app.testing = True
        client = app.test_client()

        unsupported = client.post(
            "/api/videos",
            data={"video": (io.BytesIO(b"text"), "notes.txt")},
            content_type="multipart/form-data",
        )
        empty = client.post(
            "/api/videos",
            data={"video": (io.BytesIO(b""), "empty.mp4")},
            content_type="multipart/form-data",
        )
        with mock.patch.object(
            Repository,
            "_video_metadata",
            return_value={"fps": None, "frames": None, "duration_seconds": None},
        ):
            unreadable = client.post(
                "/api/videos",
                data={"video": (io.BytesIO(b"broken"), "broken.mp4")},
                content_type="multipart/form-data",
            )

        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(unsupported.get_json()["code"], "unsupported_video_format")
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(empty.get_json()["code"], "empty_video")
        self.assertEqual(unreadable.status_code, 400)
        self.assertEqual(unreadable.get_json()["code"], "invalid_video")
        temporary_root = self.root / "input_videos" / ".uploads"
        self.assertTrue(not temporary_root.exists() or not any(temporary_root.iterdir()))

    def test_video_upload_enforces_content_type_and_size_limit(self):
        app = create_app(self.root, ImmediateProcess)
        app.testing = True
        client = app.test_client()

        wrong_type = client.post("/api/videos", json={"video": "match.mp4"})
        self.assertEqual(wrong_type.status_code, 415)
        self.assertEqual(wrong_type.get_json()["code"], "multipart_required")

        app.config["MAX_CONTENT_LENGTH"] = 32
        too_large = client.post(
            "/api/videos",
            data={"video": (io.BytesIO(b"x" * 128), "large.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(too_large.get_json()["code"], "upload_too_large")


class LauncherTests(unittest.TestCase):
    def test_windows_launcher_uses_crlf_and_ascii(self):
        launcher = Path(__file__).resolve().parents[1] / "start_gui.cmd"
        content = launcher.read_bytes()
        self.assertTrue(content)
        self.assertNotIn(b"\xef\xbb\xbf", content)
        self.assertNotIn(b"\n", content.replace(b"\r\n", b""))
        content.decode("ascii")


class SummaryIntegrationTests(unittest.TestCase):
    def test_real_summary_writes_the_complete_artifact_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracks = root / "tracks"
            output = root / "summary"
            source = root / "source.mp4"
            tracks.mkdir()
            source.write_bytes(b"source metadata placeholder")

            objects = {"ball": {}, "goalkeeper": {}, "player": {}, "referee": {}}
            positions = {
                "Red": [(100, 100), (150, 120), (200, 140)],
                "Blue": [(300, 110), (350, 130), (400, 150)],
            }
            track_id = 1
            for team, points in positions.items():
                for x, y in points:
                    objects["player"][str(track_id)] = {
                        "bbox": [x - 5, y - 20, x + 5, y],
                        "club": team,
                    }
                    track_id += 1
            keypoints = {
                str(index): [float(point[0]), float(point[1])]
                for index, point in enumerate(PitchGeometry(105, 68).vertices)
            }
            (tracks / "object_tracks.jsonl").write_text(
                json.dumps(objects) + "\n", encoding="utf-8"
            )
            (tracks / "keypoint_tracks.jsonl").write_text(
                json.dumps(keypoints) + "\n", encoding="utf-8"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "summarize_match.py"),
                    "--tracks-dir",
                    str(tracks),
                    "--output-dir",
                    str(output),
                    "--fps",
                    "1",
                    "--source",
                    str(source),
                    "--pitch-length-m",
                    "105",
                    "--pitch-width-m",
                    "68",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for filename in (
                "summary.json",
                "minute_metrics.csv",
                "team_heatmaps.png",
                "team_centres_timeline.png",
                "team_shape_timeline.png",
                "REPORT.md",
            ):
                path = output / filename
                self.assertTrue(path.is_file(), filename)
                self.assertGreater(path.stat().st_size, 0, filename)


if __name__ == "__main__":
    unittest.main()
