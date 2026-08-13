# Football Analysis

[English](README.md) | [简体中文](README.zh-CN.md)

A video-analysis pipeline for football footage. It detects and tracks players, goalkeepers, referees, and the ball, assigns teams by jersey color, maps positions to a top-down pitch, and writes an annotated video plus per-frame JSONL data.

> This repository is a maintained derivative of [Mihailo Radović's `mradovic38/football-analysis`](https://github.com/mradovic38/football-analysis).
>
> See [Credits and provenance](#credits-and-provenance) for the exact project, tutorial, dataset, and research references.

## Features

- Dedicated high-resolution, tiled YOLO ball detection plus player/official detection.
- ByteTrack-based short-term player IDs and a separate constant-velocity ball track.
- Dynamic 32-point metric-pitch calibration with RANSAC and short LK optical-flow continuation.
- Jersey-color team assignment with green-pitch masking and K-Means.
- Top-down player projection and a dynamic Voronoi control view.
- Timestamp-based robust player and ball speed estimation with a model-estimated possession overlay.
- Conservative ground/air ball kinematics with explicit uncertainty when camera intrinsics and pitch anchors are supplied.
- Native-resolution annotated MP4 plus per-frame object, keypoint, calibration, and diagnostic JSONL files.
- Optional tactical summaries, heatmaps, timelines, and quality metrics.
- Headless execution and OpenVINO model-directory support.

## Enhancements in this maintained version

Compared with the original upstream demo, this fork turns the hard-coded example into a configurable and more resilient video-analysis pipeline:

| Enhancement | What changed | Why it matters |
|---|---|---|
| Configurable CLI | Video paths, model paths, track directory, batch size, start offset, team names, four jersey colors, preview, speed, and possession can be selected without editing Python. | One checkout can analyze different matches and run in scripts. |
| Automatic model fallback | The object, keypoint, and dedicated ball detectors try an OpenVINO FP16 directory, regular OpenVINO, then PyTorch `.pt`. Explicit model options override selection. | Uses an optimized local export when available while retaining `.pt` compatibility. |
| Streaming JSONL tracks | Object and keypoint records are appended as one compact JSON value per processed frame. | Long videos no longer require reading and rewriting an ever-growing JSON array for every frame. |
| Dynamic metric pitch mapping | The supplied real pitch dimensions generate the model's 32 landmarks. RANSAC rejects outliers; LK optical flow continues an accepted camera pose for at most one second. | Pan, tilt, and zoom no longer reuse one stale projection indefinitely. |
| Dedicated ball path | Whole-frame and overlapping 2×2 tile inference feed a metric-aware temporal tracker. | Small balls are not coupled to ByteTrack's player thresholds and lost detections can be marked as short predictions. |
| Stable team assignment | The predicted team is cached for each `(object type, track ID)` pair. | Avoids running K-Means on the same short-term track every frame and reduces label flicker. |
| Independent overlays | Speed estimation and possession annotation can be enabled separately and are disabled by default. | Low-sample-rate tactical runs do not silently present unreliable instantaneous metrics. |
| Headless processing | `--no-preview` bypasses the OpenCV window. | The pipeline can run on servers and unattended machines. |
| Setup validation | `check_setup.py` reports Python, PyTorch, device and OpenCV information, checks required assets by hash, and can validate model metadata. | Missing or incompatible local assets are found before a long run. |
| Tactical reporting | `summarize_match.py` turns JSONL tracks into quality-filtered summaries, minute metrics, heatmaps, and shape timelines. | The tracking output can be inspected without writing a separate analysis program. |
| Resilience tests and repository hygiene | Empty projections, missing keypoints, ignored videos/weights/outputs, safe local API-key configuration, and placeholder directories are covered. | Local data and secrets stay out of Git, and common partial-detection cases remain testable. |

Important behavior changes from upstream:

- Detection defaults are object confidence `0.25`, dedicated ball candidate confidence `0.02`, pitch detection confidence `0.20`, and keypoint confidence `0.50`.
- The default batch size is `1`. Preview is on; speed estimation and possession annotation are off.
- Missing projections are skipped by the projection, possession, and annotation stages instead of being treated as fatal errors.
- Track IDs remain ByteTrack's short-term IDs. Cached team labels do not turn them into persistent player identities.

## Repository policy

Videos, generated outputs, model checkpoints, OpenVINO exports, local environments, and API keys are intentionally excluded from Git.

The repository contains source code, training notebooks, tests, documentation, and the small pitch image required at runtime.

## Requirements

- Python 3.11 is recommended.
- Windows, Linux, or macOS.
- A compatible object-detection checkpoint with classes in this order: `ball`, `goalkeeper`, `player`, `referee`.
- A dedicated ball-detection checkpoint whose class `0` is `ball`.
- A compatible pose checkpoint with 32 pitch keypoints.

The pinned Python dependencies are listed in [`requirements.txt`](requirements.txt). CPU inference works but can be slow. CUDA or OpenVINO may substantially improve throughput on supported hardware.

## Installation

```bash
git clone https://github.com/COSMIC-DAWN-PH/football-analysis.git
cd football-analysis
python -m venv .venv
```

Activate the environment and install dependencies.

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Linux or macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model files

Model weights are not redistributed in this repository. Train them with the included notebooks or provide compatible checkpoints at:

```text
models/weights/object-detection.pt
models/weights/keypoints-detection.pt
models/weights/ball-detection.pt
```

For each detector, automatic selection uses the first existing path in this order:

```text
models/weights/object-detection_openvino_model_fp16/
models/weights/object-detection_openvino_model/
models/weights/object-detection.pt

models/weights/keypoints-detection_openvino_model_fp16/
models/weights/keypoints-detection_openvino_model/
models/weights/keypoints-detection.pt

models/weights/ball-detection_openvino_model_fp16/
models/weights/ball-detection_openvino_model/
models/weights/ball-detection.pt
```

Check the local setup before processing a video:

```powershell
.\.venv\Scripts\python.exe check_setup.py --load-models
```

`check_setup.py` verifies the known local checkpoints by hash. A deliberate replacement may report a hash mismatch; also confirm that the object-class order and 32-keypoint layout remain compatible.

## Analyze a video

Place a video under `input_videos/` or pass any readable local path. Set the RGB colors to representative player and goalkeeper jersey colors for each team.

```powershell
.\.venv\Scripts\python.exe main.py `
  --input input_videos\match.mp4 `
  --run-dir output_videos\match `
  --pitch-length-m 105 --pitch-width-m 68 `
  --batch-size 1 `
  --club1-name Red --club1-player 220,30,30 --club1-goalkeeper 20,20,20 `
  --club2-name Blue --club2-player 30,80,220 --club2-goalkeeper 240,220,30 `
  --estimate-speed --no-annotate-possession --no-preview
```

Complete CLI:

| Option | Default | Purpose |
|---|---|---|
| `--input PATH` | required | Input match video. |
| `--output PATH` | `<run-dir>/<task-name>-analysis.mp4` | Annotated MP4 path. Parent directories are created automatically. |
| `--run-dir PATH` | `output_videos/<input-name>` | Unified folder for this run. A trailing `-input` or `_input` is removed from the task name. |
| `--object-model PATH` | automatic fallback | Object detector `.pt` file or OpenVINO model directory. |
| `--keypoints-model PATH` | automatic fallback | Pitch pose `.pt` file or OpenVINO model directory. |
| `--ball-model PATH` | automatic fallback | Dedicated ball detector `.pt` file or OpenVINO model directory. |
| `--ball-verifier-model PATH` | none | Optional second-stage ball/non-ball classifier; required by `3d` mode. |
| `--field-image PATH` | `input_videos/field_2d_v2.png` | Top-down pitch image used by the projection panel. |
| `--pitch-length-m M` | required | Measured touchline length for this specific pitch. |
| `--pitch-width-m M` | required | Measured goal-line width for this specific pitch. |
| `--tracks-dir PATH` | `<run-dir>/raw` | Directory for the three raw JSONL files. |
| `--batch-size N` | `1` | Inference batch size; must be at least 1. |
| `--skip-seconds N` | `0` | Skip the beginning of the source; cannot be negative. |
| `--estimate-speed` / `--no-estimate-speed` | disabled | Calculate and draw smoothed player and ball speed. |
| `--speed-mode off\|ground\|3d` | derived from `--estimate-speed` | Select no ball speed, ground-plane speed, or calibrated monocular 3D physics estimation. |
| `--camera-profile PATH` | none | Camera Profile JSON created by `python -m calibration camera`. |
| `--pitch-anchors PATH` | none | Pitch Anchor Set JSON created by `python -m calibration pitch`. |
| `--debug-diagnostics` | disabled | Draw the current unavailable reason; normal output hides unreliable speed. |
| `--annotate-possession` / `--no-annotate-possession` | disabled | Draw accumulated model-estimated possession. |
| `--preview` / `--no-preview` | enabled | Show or suppress the live OpenCV window. |
| `--club1-name NAME` | `Club1` | First team name. Use `Red` for compatibility with the current summary script. |
| `--club1-player R,G,B` | `232,247,248` | First team's representative outfield jersey color. |
| `--club1-goalkeeper R,G,B` | `6,25,21` | First team's representative goalkeeper color. |
| `--club2-name NAME` | `Club2` | Second team name. Use `Blue` for compatibility with the current summary script. |
| `--club2-player R,G,B` | `172,251,145` | Second team's representative outfield jersey color. |
| `--club2-goalkeeper R,G,B` | `239,156,132` | Second team's representative goalkeeper color. |

RGB arguments must contain exactly three integers from `0` to `255`.

Run `python main.py --help` for the complete interface.

### Reliable ball-speed modes

`--speed-mode off|ground|3d` is the authoritative ball-speed switch. The older
`--estimate-speed` option remains an alias for `ground` mode. `3d` mode requires
a Camera Profile, a Pitch Anchor Set, a ball/non-ball verifier, and formally
promoted dynamic 1280 ball/keypoint exports; it fails fast instead of silently
falling back to a homography-based value.

Create the calibration artifacts with:

```powershell
.\.venv\Scripts\python.exe -m calibration camera --images "checkerboard\*.jpg" --columns 9 --rows 6 --square-size-m 0.024 --profile-id camera-1 --output calibration-data\camera_profile.json
.\.venv\Scripts\python.exe -m calibration pitch --video input_videos\match.mp4 --frame 0 --pitch-points "0,0;52.5,0;105,0;105,68;52.5,68;0,68" --output calibration-data\pitch_anchors.json
```

Project-specific capture and rollout checklists may be kept locally under
`plan/`; that directory is intentionally excluded from release commits.

## Outputs

The main pipeline produces:

- An annotated video at `<run-dir>/<task-name>-analysis.mp4`.
- `<run-dir>/raw/object_tracks.jsonl`.
- `<run-dir>/raw/keypoint_tracks.jsonl`.
- `<run-dir>/raw/calibration_tracks.jsonl`.
- `<run-dir>/raw/diagnostics.jsonl`.
- `<run-dir>/raw/quality_summary.json`.

By default, each run groups its annotated video, `raw` tracks, and `summary` artifacts in one task folder. `--output` and `--tracks-dir` remain available as explicit overrides.

Each JSONL line represents one processed frame. Object records may include `bbox`, `confidence`, `position_m`, `projection`, `club`, `club_color`, `has_ball`, and `speed`. Ball records also expose `observed`, `track_confidence`, `track_segment`, `track_confirmed`, `track_state`, `speed_state`, and `speed_reason`. Ball `speed` is emitted only when a confirmed segment passes detection-confidence, calibration-quality, trajectory-observability, and uncertainty checks. Normal video hides unavailable values; `--debug-diagnostics` draws the exact reason such as `pose_stale`, `track_ambiguous`, or `uncertainty_high`.

`object_tracks.jsonl` has four top-level object groups. JSON serialization turns numeric track IDs into strings:

```json
{"ball":{"1":{"bbox":[915.2,511.0,928.4,525.7],"confidence":0.72,"position_m":[52.1,31.4],"observed":true,"track_confidence":0.72}},"goalkeeper":{},"player":{"12":{"bbox":[804.1,392.5,858.8,556.2],"confidence":0.91,"position_m":[48.2,35.7],"projection":[244.9,181.3],"club":"Red","speed":18.4}},"referee":{}}
```

`position_m` and `projection` are omitted when calibration is not usable or the projected point is outside the pitch. Invalid speed is omitted rather than clamped. A predicted ball has `observed:false` and expires after 0.5 seconds. Each `keypoint_tracks.jsonl` line maps landmark indexes to image coordinates. Each calibration line records the timestamp, status, keypoint/inlier counts, reprojection error, coverage spans, flow quality, and calibration age.

The old upstream `object_tracks.json` array is a legacy format. New output uses JSON Lines: consume the file one line at a time rather than loading it as one JSON array. Starting a new run in the same track directory removes previous generated JSONL files.

The video combines the source frame with IDs, team-colored annotations, pitch keypoints, and a top-down pitch projection. Valid player speed remains visible for about one second. Ball speed is shown only on frames that pass the trust gates. Unassigned possession is labeled `Unconfirmed`. Speed and possession graphics are controlled independently.

The generated video keeps the input frame dimensions and aspect ratio. Frames are encoded as MP4 after processing; the source audio track is not copied.

## Optional tactical summary

After generating tracks, `summarize_match.py` can create quality-filtered team metrics and visualizations.

This optional script also requires Matplotlib: `python -m pip install matplotlib`.

```powershell
.\.venv\Scripts\python.exe summarize_match.py `
  --tracks-dir output_videos\match\raw `
  --fps 1 `
  --source input_videos\match.mp4 `
  --pitch-length-m 105 --pitch-width-m 68
```

When `--output-dir` is omitted, the summary is written to `output_videos\match\summary`; pass the option explicitly to override it.

`--fps` is the sampling rate represented by the JSONL lines, not automatically the original video's frame rate. For example, tracks produced from one sampled frame per source second require `--fps 1`; normal full-frame processing usually uses the source FPS.

The current summary recognizes team names `Red` and `Blue`, so use those exact names when tracks will be summarized. It prefers metric `position_m` and `calibration_tracks.jsonl`; old object/keypoint JSONL remains readable through legacy per-frame RANSAC projection.

A detected calibration needs at least 6 keypoints, 5 RANSAC inliers, a 60% inlier ratio, median reprojection error no greater than 5 px, and coverage of at least 30% of pitch length and 25% of width. Flow can continue it for up to one second with decaying quality. Positions outside the supplied metric pitch are not written.

The report directory contains:

- `summary.json`: source metadata, sample rate, accepted-frame quality rate, team aggregates, and the most separated/compressed minutes.
- `minute_metrics.csv`: per-minute visible counts, centres, width, length, compactness, and inter-team centre separation.
- `team_heatmaps.png`: quality-filtered spatial density for `Red` and `Blue`.
- `team_centres_timeline.png`: longitudinal team centres by minute in fixed pitch coordinates.
- `team_shape_timeline.png`: team width and length by minute.
- `REPORT.md`: a readable Chinese interpretation of the calculated sample.

These summaries are observational estimates from a single moving camera. They are not equivalent to calibrated multi-camera or GPS tracking data. Low-frame-rate samples are suitable for broad team-shape trends, not running distance, sprints, or instantaneous speed.

## Training

The repository includes the older notebooks plus a reproducible local/Roboflow dataset workflow:

- [`models/object_detection_train.ipynb`](models/object_detection_train.ipynb)
- [`models/keypoints_detection_train.ipynb`](models/keypoints_detection_train.ipynb)
- [`training/README.md`](training/README.md)

The new tools extract 800 ball and 400 pitch annotation candidates, validate split leakage and the 32-point flip order, fine-tune YOLO11s/keypoint weights on cloud CUDA, and export OpenVINO FP16. They accept local YOLO datasets or an unpacked Roboflow YOLO export. Weights are never promoted automatically.

The notebooks directly reference two datasets published by **Mihailo** on Roboflow Universe:

- [football-players-detection, version 2](https://universe.roboflow.com/mihailo/football-players-detection-3zvbc-7ocfe/dataset/2) — object detection, CC BY 4.0.
- [football-field-detection, version 1](https://universe.roboflow.com/mihailo/football-field-detection-f07vi-apxzb/dataset/1) — 32-point keypoint detection, CC BY 4.0.

Review the dataset pages and their current terms before downloading, training, or redistributing derived weights.

## Limitations

- IDs are short-term tracker IDs, not player identities or jersey numbers.
- A moving single camera still has incomplete pitch coverage; projection and speed are omitted when calibration is stale or weak.
- Small-ball detection is difficult and can affect possession estimates.
- Speed is a robust fit over timestamped metric positions. Values above 45 km/h or with implausible acceleration are rejected, not capped.
- Team assignment depends on representative colors and can be confused by lighting, bibs, spectators, or similar kits.
- Model checkpoints and their licenses are the user's responsibility.

## Credits and provenance

### Direct upstream

This codebase is directly based on [Mihailo Radović's `mradovic38/football-analysis`](https://github.com/mradovic38/football-analysis).

The upstream project supplied the initial architecture for tracking, keypoints, team assignment, homography, projection, Voronoi display, possession, and speed estimation.

The bundled `input_videos/field_2d_v2.png` pitch image also comes from that upstream repository. Its MIT copyright notice is preserved in [`LICENSE`](LICENSE).

This fork is maintained at [`COSMIC-DAWN-PH/football-analysis`](https://github.com/COSMIC-DAWN-PH/football-analysis).

It adds a configurable CLI, safer long-video handling, JSONL writing, OpenVINO-aware model selection, headless execution, setup checks, resilience fixes, tests, and optional tactical summaries.

### Tutorials and technical references

The upstream project identifies the following materials as inspiration or technical references. They are credited here for traceability; listing them does not claim that this fork copied code directly from every source.

- [Code In a Jiffy — Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python](https://www.youtube.com/watch?v=neBZ6huolkg)
- [Roboflow — Football AI Tutorial: From Basics to Advanced Stats with Python](https://www.youtube.com/watch?v=aBVGKoNZQUw)
- [Roboflow Football AI notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/football-ai.ipynb)
- [DonsetPG/Narya](https://github.com/DonsetPG/narya) — football tracking and homography work.
- [PiotrGrabysz/PitchGeometry](https://github.com/PiotrGrabysz/PitchGeometry) — football-pitch keypoint detection.
- R. Alhejaily et al., [“Automatic Team Assignment and Jersey Number Recognition in Football Videos”](https://doi.org/10.32604/iasc.2023.033062), 2023.

Core runtime libraries include [Ultralytics](https://github.com/ultralytics/ultralytics), [Supervision](https://github.com/roboflow/supervision), OpenCV, PyTorch, SciPy, scikit-learn, and pandas. Their respective authors and licenses apply.

## License

The repository source is distributed under the [MIT License](LICENSE), preserving the upstream copyright notice.

Ultralytics software and YOLO models have separate licensing terms, including AGPL-3.0 and commercial options.

Roboflow datasets and independently obtained checkpoints retain their own licenses. Review all applicable terms for your use case.
