# XbotGo fine-tuning workflow

Use at least four pitches and keep entire videos/pitches in one split. Split approximately 70/15/15, with at least one completely unseen pitch in test; adjacent frames must never cross splits. The first-round target is 800 ball frames (at least 600 ball-positive plus empty scenes and hard negatives) and 400 pitch-keypoint frames covering pan, zoom, backlight, close, and wide views.

Extract annotation candidates locally:

```powershell
python -m training.prepare_dataset extract --task ball --videos video1.mp4 video2.mp4 --output-dir training-data/ball --model models/weights/ball-detection.pt
python -m training.prepare_dataset extract --task pitch --videos video1.mp4 video2.mp4 --output-dir training-data/pitch --model models/weights/keypoints-detection.pt
```

Correct every prelabel in a YOLO-compatible annotation tool. For pitch data, add this exact permutation to `data.yaml` so horizontal augmentation preserves landmark identities:

When creating the final splits, keep/update `manifest.csv` with `split` and `pitch` columns. The validator uses `source_video` and `pitch` to reject cross-split leakage and verifies that test includes an unseen pitch.

```yaml
kpt_shape: [32, 3]
flip_idx: [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]
```

Roboflow private-project exports are supported without binding training to its API:

```powershell
python -m training.prepare_dataset unpack --zip roboflow-export.zip --output-dir training-data/ball-yolo
```

Validate before upload/training:

```powershell
python -m training.prepare_dataset validate --task ball --data training-data/ball-yolo/data.yaml
python -m training.prepare_dataset validate --task pitch --data training-data/pitch-yolo/data.yaml
```

Run on a CUDA cloud machine from the repository root:

```bash
python -m training.train_models --task ball --data training-data/ball-yolo/data.yaml --device 0
python -m training.train_models --task pitch --data training-data/pitch-yolo/data.yaml --device 0 --batch 4
```

The script writes held-out metrics and a dynamic-shape OpenVINO FP16 export, allowing 1920 whole-frame and 1280 tile inference, but deliberately does not replace runtime weights. Promote a ball model only after it reaches at least 75% visible-ball recall and 90% precision on unseen-pitch videos, then rerun the end-to-end calibration and speed checks.
