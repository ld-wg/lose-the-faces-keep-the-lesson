#!/usr/bin/env python3
"""
Simple Webcam Face Detection
============================
Run trained YOLOv8 model on webcam feed.

Usage:
    python webcam.py                    # Use latest model
    python webcam.py --model custom.pt  # Use specific model
    python webcam.py --conf 0.5        # Higher confidence threshold

Controls:
    Press 'q' to quit
    Press 'c' to toggle confidence display
"""

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


def find_latest_model():
    """Find the latest trained model."""
    runs_dir = Path("runs/train")

    if not runs_dir.exists():
        print("❌ No training runs found in runs/train/")
        return None

    # Find all best.pt files and get the most recent
    model_files = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            best_pt = run_dir / "weights" / "best.pt"
            if best_pt.exists():
                model_files.append((best_pt.stat().st_mtime, best_pt))

    if not model_files:
        print("❌ No trained models found in runs/train/")
        return None

    # Get most recent model
    model_files.sort(key=lambda x: x[0], reverse=True)
    latest_model = model_files[0][1]

    print(f"✅ Using latest model: {latest_model}")
    return latest_model


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam Face Detection")
    parser.add_argument('--model', type=str, help='Path to model file (auto-detect if not provided)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, help='Device to use (mps, cuda, cpu). Default: auto-detect')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
            print("🖥️ Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
            print("🖥️ Using NVIDIA GPU")
        else:
            device = "cpu"  # CPU only
            print("🖥️ Using CPU")
    else:
        device = args.device
        print(f"🖥️ Using specified device: {device}")

    # Find model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
    else:
        model_path = find_latest_model()
        if not model_path:
            return

    print(f"🚀 Loading model: {model_path}")

    # Load model
    model = YOLO(str(model_path))
    model.to(device)

    # Open webcam
    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✅ Webcam ready! Press 'q' to quit, 'c' to toggle confidence scores")
    print("💡 Tip: Lower confidence threshold if no faces detected")

    show_conf = True  # Show confidence scores
    show_debug = False  # Show debug info

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame")
            break

        # Run detection
        results = model(frame, conf=args.conf, device=device, verbose=False)

        # Get detection info for debugging
        detections = results[0]
        num_detections = len(detections.boxes)

        # Draw results
        annotated_frame = results[0].plot(conf=show_conf)

        # Always show face count at the top
        face_text = f"Faces: {num_detections}"
        cv2.putText(annotated_frame, face_text, (10, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        # Add debug overlay
        if show_debug:
            # Add additional debug info
            info_text = f"Conf Threshold: {args.conf:.2f}"
            cv2.putText(annotated_frame, info_text, (10, 70),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)
        
        if num_detections == 0:
            cv2.putText(annotated_frame, "No faces detected - try lowering confidence",
                       (10, 70 if not show_debug else 105), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("YOLOv8 Face Detection", annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            show_conf = not show_conf
            status = "ON" if show_conf else "OFF"
            print(f"Confidence display: {status}")
        elif key == ord('d'):
            show_debug = not show_debug
            status = "ON" if show_debug else "OFF"
            print(f"Debug info: {status}")
        elif key == ord('['):
            args.conf = max(0.01, args.conf - 0.05)
            print(f"Confidence threshold: {args.conf:.2f}")
        elif key == ord(']'):
            args.conf = min(0.99, args.conf + 0.05)
            print(f"Confidence threshold: {args.conf:.2f}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed")


if __name__ == "__main__":
    main()
