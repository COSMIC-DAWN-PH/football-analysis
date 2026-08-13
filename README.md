# Football Analysis

[English](README.md) | [简体中文](README.zh-CN.md)

A video-analysis pipeline for football footage. It detects and tracks players, goalkeepers, referees, and the ball, assigns teams by jersey color, maps positions to a top-down pitch, and writes an annotated video plus per-frame JSONL data.

> This repository is a maintained derivative of [Mihailo Radović's `mradovic38/football-analysis`](https://github.com/mradovic38/football-analysis).
>
> See [Credits and provenance](#credits-and-provenance) for the exact project, tutorial, dataset, and research references.

## Features

- YOLO-based detection for the ball, players, goalkeepers, and referees.
- ByteTrack-based short-term object IDs.
- 32-point football-pitch keypoint detection and homography mapping.
- Jersey-color team assignment with green-pitch masking and K-Means.
- Top-down player projection and a dynamic Voronoi control view.
- Optional player-speed estimation and model-estimated possession overlay.
- Annotated MP4 output and per-frame object/keypoint JSONL files.
- Optional tactical summaries, heatmaps, timelines, and quality metrics.
- Headless execution and OpenVINO model-directory support.

## Repository policy

Videos, generated outputs, model checkpoints, OpenVINO exports, local environments, and API keys are intentionally excluded from Git.

The repository contains source code, training notebooks, tests, documentation, and the small pitch image required at runtime.

## Requirements

- Python 3.11 is recommended.
- Windows, Linux, or macOS.
- A compatible object-detection checkpoint with classes in this order: `ball`, `goalkeeper`, `player`, `referee`.
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
```

The application prefers these OpenVINO directories when they exist:

```text
models/weights/object-detection_openvino_model_fp16/
models/weights/keypoints-detection_openvino_model_fp16/
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
  --output output_videos\match-analysis.mp4 `
  --tracks-dir output_videos\match-tracks `
  --batch-size 1 `
  --club1-name Red --club1-player 220,30,30 --club1-goalkeeper 20,20,20 `
  --club2-name Blue --club2-player 30,80,220 --club2-goalkeeper 240,220,30 `
  --estimate-speed --no-annotate-possession --no-preview
```

Important options:

| Option | Purpose |
|---|---|
| `--estimate-speed` | Adds smoothed speed estimates to annotations and object JSONL. |
| `--annotate-possession` | Draws the accumulated model-estimated possession bar. |
| `--no-preview` | Disables the OpenCV window for headless processing. |
| `--skip-seconds N` | Skips the first `N` seconds. |
| `--batch-size N` | Controls inference batch size; use `1` on constrained CPUs. |
| `--object-model PATH` | Overrides the object model or OpenVINO directory. |
| `--keypoints-model PATH` | Overrides the pitch-keypoint model or OpenVINO directory. |

Run `python main.py --help` for the complete interface.

## Outputs

The main pipeline produces:

- An annotated video at the path passed to `--output`.
- `object_tracks.jsonl` under `--tracks-dir`.
- `keypoint_tracks.jsonl` under `--tracks-dir`.

Each JSONL line represents one processed frame. Object records may include `bbox`, `club`, `club_color`, `projection`, `has_ball`, and `speed`, depending on the object type and enabled analysis.

The video combines the source frame with IDs, team-colored annotations, pitch keypoints, and a top-down pitch projection. Speed and possession graphics are controlled independently.

## Optional tactical summary

After generating tracks, `summarize_match.py` can create quality-filtered team metrics and visualizations.

This optional script also requires Matplotlib: `python -m pip install matplotlib`.

```powershell
.\.venv\Scripts\python.exe summarize_match.py `
  --tracks-dir output_videos\match-tracks `
  --output-dir output_videos\match-summary `
  --fps 30 `
  --source input_videos\match.mp4
```

These summaries are observational estimates from a single moving camera. They are not equivalent to calibrated multi-camera or GPS tracking data.

## Training

The repository includes:

- [`models/object_detection_train.ipynb`](models/object_detection_train.ipynb)
- [`models/keypoints_detection_train.ipynb`](models/keypoints_detection_train.ipynb)

Training uses Roboflow downloads and therefore requires your own API key. Copy `config.example.py` to `config.py`, add the key locally, and never commit it.

The notebooks directly reference two datasets published by **Mihailo** on Roboflow Universe:

- [football-players-detection, version 2](https://universe.roboflow.com/mihailo/football-players-detection-3zvbc-7ocfe/dataset/2) — object detection, CC BY 4.0.
- [football-field-detection, version 1](https://universe.roboflow.com/mihailo/football-field-detection-f07vi-apxzb/dataset/1) — 32-point keypoint detection, CC BY 4.0.

Review the dataset pages and their current terms before downloading, training, or redistributing derived weights.

## Limitations

- IDs are short-term tracker IDs, not player identities or jersey numbers.
- A moving single camera introduces projection error and incomplete pitch coverage.
- Small-ball detection is difficult and can affect possession estimates.
- Speed is inferred from frame-to-frame projected positions and may reach the configured cap after tracking or homography jumps.
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
