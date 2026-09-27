"""
Train YOUR OWN anomaly model on YOUR normal CCTV footage.
No manual labeling: YOLO + ByteTrack auto-extracts normal trajectory features.

Usage:
  # 1) put 5-10 min NORMAL-only video (no running/fighting) here:
  python scripts/train_ml_anomaly.py --source data/my_normal.mp4 --weights yolo11s.pt
  # or image folder:
  python scripts/train_ml_anomaly.py --source data/MOT17-YOLO/images/train --weights yolo11s.pt --limit 400
  # output: models/ml_anomaly.pkl
"""
import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from ultralytics import YOLO
from src.tracking.tracker import TrackerWrapper
from src.anomaly.ml_model import extract_features


def collect_features(source, weights="yolo11s.pt", conf=0.35, limit=800):
    model = YOLO(weights)
    tracker = TrackerWrapper(tracker_type="bytetrack")
    X = []
    frame_idx = 0
    prev_area = {}

    def handle_frame(frame):
        nonlocal frame_idx
        res = model.predict(frame, conf=conf, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                if int(b.cls) == 0 or model.names[int(b.cls)] == "person":
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    dets.append([x1, y1, x2, y2, float(b.conf), int(b.cls)])
        dets = np.array(dets) if dets else np.zeros((0, 6))
        tracks = tracker.update(dets, frame, frame_idx=frame_idx)
        for x1, y1, x2, y2, tid, cf in tracks:
            v = tracker.analyzer.velocity(tid)
            a = tracker.analyzer.direction(tid)
            flow = tracker.analyzer.flow_angle
            adiff = 0.0
            if a is not None and flow is not None:
                d = abs(a - flow)
                adiff = min(d, 360 - d)
            dwell = frame_idx - tracker.analyzer.last_seen.get(tid, frame_idx)
            # note: last_seen == current frame, use track_birth approx via hist len
            dwell = len(tracker.analyzer.hist.get(tid, []))
            w, h = (x2 - x1), (y2 - y1)
            area = w * h
            prev = prev_area.get(tid, area)
            area_change = abs(area - prev) / max(prev, 1)
            prev_area[tid] = area
            if len(tracker.analyzer.hist.get(tid, [])) >= 5:
                X.append(extract_features(v, adiff, dwell, w, h, area_change))
        frame_idx += 1
        return len(X)

    p = Path(source)
    if p.is_dir():
        imgs = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        for imgp in imgs[:limit]:
            frame = cv2.imread(str(imgp))
            if frame is None:
                continue
            n = handle_frame(frame)
            if n >= limit:
                break
    else:
        cap = cv2.VideoCapture(str(p) if p.exists() else source)
        while cap.isOpened() and len(X) < limit:
            ret, frame = cap.read()
            if not ret:
                break
            handle_frame(frame)
        cap.release()
    print(f"[TrainML] collected {len(X)} normal samples from {frame_idx} frames")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="normal-only video or image folder")
    ap.add_argument("--weights", type=str, default="yolo11s.pt")
    ap.add_argument("--limit", type=int, default=800, help="max feature samples")
    ap.add_argument("--save", type=str, default="models/ml_anomaly.pkl")
    ap.add_argument("--contamination", type=float, default=0.05)
    args = ap.parse_args()

    from src.anomaly.ml_model import TrajectoryMLDetector
    X = collect_features(args.source, args.weights, limit=args.limit)
    if len(X) < 50:
        print(f"Too few samples ({len(X)}). Use longer normal clip.")
        return
    det = TrajectoryMLDetector(contamination=args.contamination)
    det.train(X, save_path=args.save)


if __name__ == "__main__":
    main()
