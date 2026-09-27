"""
Train CAE on Avenue normal frames (if dataset present). Else demo on MOT17.
Usage: python scripts/train_cae.py --epochs 15
"""
import argparse, cv2
from pathlib import Path
import numpy as np
from src.anomaly.cae import AnomalyCAE

def load_frames(root, limit=2000):
    root=Path(root)
    frames=[]
    for p in sorted(root.rglob("*.jpg"))[:limit]:
        img=cv2.imread(str(p))
        if img is not None: frames.append(img)
    for p in sorted(root.rglob("*.png"))[:limit-len(frames)]:
        img=cv2.imread(str(p))
        if img is not None: frames.append(img)
    # also try Avenue videos if present
    for v in sorted(root.rglob("*.avi"))[:5]:
        cap=cv2.VideoCapture(str(v))
        while len(frames)<limit and cap.isOpened():
            ret,f=cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
    return frames

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--data", type=str, default="data/avenue/training_videos")
    ap.add_argument("--save", type=str, default="models/cae_avenue.pt")
    args=ap.parse_args()
    data=Path(args.data)
    if not data.exists():
        # fallback to MOT17 train as proxy normal data
        data=Path("data/MOT17-YOLO/images/train")
        print(f"[CAE] Avenue not found, using {data} as proxy")
    frames=load_frames(data, limit=1500)
    if len(frames)<50:
        print(f"[CAE] not enough frames {len(frames)} at {data}")
        return
    print(f"[CAE] training on {len(frames)} frames for {args.epochs} epochs")
    cae=AnomalyCAE()
    thresh=cae.train(frames, epochs=args.epochs, save_path=args.save)
    print(f"[CAE] done thresh {thresh:.5f} -> {args.save}")
    # quick test
    dummy=frames[0]
    mse, is_anom = cae.score(dummy)
    print(f"[CAE] test mse {mse:.5f} anom {is_anom}")

if __name__=="__main__":
    main()
