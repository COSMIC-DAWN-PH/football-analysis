import argparse
import hashlib
import platform
import sys
from pathlib import Path

import cv2
import torch


EXPECTED_FILES = {
    Path("models/weights/object-detection.pt"): (
        "69c652bfa9814ef882c439617f04b8fd5749b6b8455aaa3c36110bc2e802aadd"
    ),
    Path("models/weights/keypoints-detection.pt"): (
        "06623b51f77f51695cde731da146596e6df73c95a5b4776f6afe7094389ed209"
    ),
    Path("input_videos/field_2d_v2.png"): (
        "9f37bca64f4cd471181962403282ba7356015cfce9a63817b3185353fc7aaa39"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local Football-Analysis setup.")
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Also load the runtime YOLO checkpoints and validate their metadata",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"OpenCV: {cv2.__version__}")

    ok = True
    all_present = True
    for path, expected_hash in EXPECTED_FILES.items():
        if not path.is_file():
            print(f"MISSING: {path}")
            ok = False
            all_present = False
            continue
        actual_hash = _sha256(path)
        status = "OK" if actual_hash == expected_hash else "HASH MISMATCH"
        print(f"{status}: {path}")
        ok &= actual_hash == expected_hash

    if args.load_models and all_present:
        from ultralytics import YOLO

        object_model = YOLO("models/weights/object-detection.pt")
        object_names = [object_model.names[index] for index in sorted(object_model.names)]
        expected_names = ["ball", "goalkeeper", "player", "referee"]
        if object_names != expected_names:
            print(f"INCOMPATIBLE object classes: {object_names}")
            ok = False
        else:
            print(f"OK: object classes {object_names}")

        keypoints_model = YOLO("models/weights/keypoints-detection.pt")
        keypoint_shape = keypoints_model.model.yaml.get("kpt_shape")
        if not keypoint_shape or keypoint_shape[0] != 32:
            print(f"INCOMPATIBLE keypoint shape: {keypoint_shape}")
            ok = False
        else:
            print(f"OK: keypoint shape {keypoint_shape}")

        ball_candidates = (
            Path("models/weights/ball-detection_openvino_model_fp16"),
            Path("models/weights/ball-detection_openvino_model"),
            Path("models/weights/ball-detection.pt"),
        )
        ball_path = next((path for path in ball_candidates if path.exists()), None)
        if ball_path is None:
            print("MISSING: dedicated ball model")
            ok = False
        else:
            ball_model = YOLO(str(ball_path), task="detect")
            ball_names = [ball_model.names[index] for index in sorted(ball_model.names)]
            if ball_names != ["ball"]:
                print(f"INCOMPATIBLE ball classes: {ball_names}")
                ok = False
            else:
                print(f"OK: ball classes {ball_names} ({ball_path})")

    print("Setup is ready." if ok else "Setup needs attention.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
