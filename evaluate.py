"""Evaluate detector + anomaly on MOT17 val with enhanced report"""
from ultralytics import YOLO
from pathlib import Path
import time

def eval_detector():
    for w in ["models/yolo11s_mot17.pt", "yolo11s.pt", "yolo11n.pt"]:
        p=Path(w)
        if p.exists():
            print(f"\n=== {w} ===")
            t0=time.time()
            model=YOLO(str(p))
            metrics=model.val(data="configs/mot17.yaml", imgsz=480, verbose=False)
            dt=time.time()-t0
            print(f"mAP50 {metrics.box.map50:.4f} mAP50-95 {metrics.box.map:.4f} P {metrics.box.mp:.4f} R {metrics.box.mr:.4f} time {dt:.1f}s")
            # FPS estimate
            import cv2, numpy as np
            dummy=np.zeros((480,640,3), dtype=np.uint8)
            import time as tm
            s=tm.time()
            for _ in range(20): model.predict(dummy, verbose=False, imgsz=480)
            fps=20/(tm.time()-s)
            print(f"FPS est {fps:.1f} on GTX 1650")
            break
    print("\nAnomaly rules: vel floor 14, dir only if >9px/f, persist 4-5, cooldown 45, loiter disabled for webcam")

if __name__=="__main__":
    eval_detector()
