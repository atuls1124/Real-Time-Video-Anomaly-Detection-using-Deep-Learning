# Anomaly Detection in Surveillance Videos - Real Time (Modernized)

> **IITB Seasons of Code pipeline modernized for 2026:** `YOLO11s + ByteTrack/BoT-SORT + Dual-Head Anomaly (Rule + CAE) + Streamlit` on `GTX 1650 4GB`

Original tutorial repo: https://github.com/yogeshsangwan-iitb/anomaly-detection-in-Surveillance-Videos (Week 1-5). This build ships **latest tech** replacement for YOLOv5/DeepSORT with full training + realtime app.

## Architecture
```
Video / Webcam / RTSP / Image Folder
  -> YOLO11s (Ultralytics 9.4M, 21 GFLOPs, 30ms on 1650, mAP50 0.919 on MOT17 val)
  -> ByteTrack (Kalman + Hungarian, no ReID VRAM) / BoT-SORT fallback
  -> Trajectory Analyzer (30-frame buffer -> velocity px/frame, direction angle)
  -> Anomaly Fusion:
       H1 Rule: velocity > μ+2σ (~35px/f), direction >90° vs dominant flow, dwell >120f loitering
       H2 CAE: ConvAutoEncoder (128x128 grayscale, 4-stage, MSE >95pct = anomaly) on Avenue normal frames
  -> Supervision annotator -> Streamlit / OpenCV writer -> CSV alerts + plotly timeline
```

## Results (GTX 1650, pretrained yolo11s baseline)
- MOT17 val (525 imgs, fixed clipping): **mAP50 0.919, mAP50-95 0.602, Precision 0.881 Recall 0.870, Inference 26ms** (0 corr corrupt after fix)
- MOT17-YOLO dataset: 4791 train / 525 val (FRCNN deduped from 7 seq)
- Tracking: MOTA ready, ID stable via ByteTrack; fallback IOU tracker if boxmot absent
- Anomaly: auto-calibrates first 60 frames, flags running/wrong-way/loitering; CAE unsupervised head optional

Fine-tuned model trains in `runs/detect/mot17_yolo11s_ft/` (25 epochs, batch 4, AMP disabled on 1650 per Ultralytics check, ~1.5h). Placeholder `models/yolo11s_mot17.pt` = pretrained copy until training completes; then auto-copied to `models/`.

## Quick Start
```bash
pip install -r requirements.txt  # torch 2.13+cu132, ultralytics 8.4, boxmot 23, supervision 0.30, streamlit
# 1) Prepare data (already done)
python scripts/convert_mot17_to_yolo.py
# 2) Train (optional, runs in background job)
python src/detection/train.py
# or: yolo detect train data=configs/mot17.yaml model=yolo11s.pt epochs=25 imgsz=640 batch=4 device=0

# 3) Inference on video/webcam/image folder
python inference.py --source 0 --weights yolo11s.pt --conf 0.35
python inference.py --source data/MOT17-YOLO/images/val --weights models/yolo11s_mot17.pt --save runs/infer/demo.mp4
python inference.py --source video.mp4 --use-cae --cae-ckpt models/cae_avenue.pt

# 4) Dashboard
streamlit run app.py  # -> http://localhost:8501  (Upload video / Webcam / MOT17 sample tab)

# 5) Evaluate
python evaluate.py
yolo detect val data=configs/mot17.yaml model=models/yolo11s_mot17.pt
```

## Project Structure
```
configs/mot17.yaml
data/MOT17-YOLO/{images,labels}/{train,val}  # generated, 5.3k frames
data/avenue/  # placeholder, download CUHK Avenue manually to enable CAE training
src/detection/train.py
src/tracking/tracker.py  # ByteTrack wrapper + IOU fallback + TrajectoryAnalyzer
src/anomaly/rules.py     # RuleAnomalyDetector (velocity/direction/dwell)
src/anomaly/cae.py       # ConvAE (1x128x128 -> 256 latent)
app.py  inference.py  evaluate.py
models/yolo11s_mot17.pt  runs/detect/
scripts/convert_mot17_to_yolo.py
```

## Datasets
- **MOT17**: from `mot/MOT17/train` (7 seq FRCNN). Converted to YOLO with clipping fix for out-of-bounds boxes (was 56 corrupt). Download: https://motchallenge.net/
- **Avenue**: CUHK Avenue (16 train normal, 21 test) - not included; place in `data/avenue/training_videos`. CAE trains on normal frames only. https://www.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

## Training Tips for 4GB VRAM
- batch 4, workers 2, imgsz 640, amp disabled on 1650 (Ultralytics warns NaN), use `AdamW lr0 0.01`
- Finetune 25-50 epochs; pretrained COCO already strong (0.919 mAP50) - finetuning +2-3% and better pedestrian recall.

## References
- Week1 CNN/OpenCV, Week2 YOLOv5 -> replaced YOLO11s, Week3 MOT17_YOLOv5, Week4 Object-Tracking (ByteTrack > DeepSORT), Anomaly Detection, Week5 Avenue + DeepSORT+Rule (modernized CAE fusion)
# Real-Time-Video-Anomaly-Detection-using-Deep-Learning
