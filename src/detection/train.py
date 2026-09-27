"""Train YOLO11s on MOT17-YOLO - optimized for GTX 1650 4GB"""
from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_YAML = BASE / "configs" / "mot17.yaml"

def train():
    model = YOLO("yolo11s.pt")  # will auto-download 9.4M params
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,  # 50 is enough for 5k images on 1650, can bump to 80 if time
        imgsz=640,
        batch=4,  # 4GB VRAM safe, 6 may OOM with workers
        workers=2,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        mosaic=0.8,
        mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        amp=True,
        cache=False,
        project=str(BASE / "runs" / "detect"),
        name="mot17_yolo11s",
        exist_ok=True,
        save_period=10,
        val=True,
        plots=True,
    )
    # validate and export
    metrics = model.val()
    print(metrics)
    # copy best to models/
    import shutil
    best = BASE / "runs" / "detect" / "mot17_yolo11s" / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, BASE / "models" / "yolo11s_mot17.pt")
        print(f"Saved {BASE / 'models' / 'yolo11s_mot17.pt'}")
    return results

if __name__ == "__main__":
    train()
