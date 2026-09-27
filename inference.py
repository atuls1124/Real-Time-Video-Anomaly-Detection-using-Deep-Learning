"""
Real-time inference: YOLO11 + ByteTrack + RuleAnomaly (+ optional CAE)
Usage:
  python inference.py --source 0  (webcam)
  python inference.py --source mot/MOT17/train/MOT17-02-FRCNN/img1 --weights models/yolo11s_mot17.pt
  python inference.py --source video.mp4 --save runs/infer/output.mp4
"""
import argparse, cv2, time, csv
from pathlib import Path
import numpy as np

from ultralytics import YOLO
from src.tracking.tracker import TrackerWrapper
from src.anomaly.rules import RuleAnomalyDetector
try:
    from src.anomaly.cae import AnomalyCAE
    HAS_CAE=True
except: HAS_CAE=False
try:
    from src.anomaly.ml_model import TrajectoryMLDetector, extract_features
    HAS_ML=True
except: HAS_ML=False

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="0 webcam or path to video/images folder")
    p.add_argument("--weights", type=str, default="yolo11s.pt", help="yolo weights (auto-download if yolo11s.pt) or models/yolo11s_mot17.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--use-cae", action="store_true")
    p.add_argument("--cae-ckpt", type=str, default="models/cae_avenue.pt")
    p.add_argument("--use-ml", action="store_true", help="fuse own IsolationForest trajectory model")
    p.add_argument("--ml-ckpt", type=str, default="models/ml_anomaly.pkl")
    return p.parse_args()

def get_capture(source):
    if source=="0": return cv2.VideoCapture(0)
    p=Path(source)
    if p.is_dir():
        # image folder -> glob
        imgs=sorted(p.glob("*.jpg"))+sorted(p.glob("*.png"))
        return imgs  # handle as list
    return cv2.VideoCapture(str(p))

def main():
    args=parse_args()
    w=str(Path(args.weights))
    if not Path(w).exists() and w.endswith(".pt") and not w.startswith("yolo"):
        # fallback to pretrained
        print(f"Weights {w} not found, using yolo11s.pt pretrained")
        w="yolo11s.pt"
    model=YOLO(w)
    # autobenchmark device
    tracker=TrackerWrapper(tracker_type="bytetrack")
    detector_rule=RuleAnomalyDetector(auto_calib_frames=60)
    cae=None
    if args.use_cae and HAS_CAE and Path(args.cae_ckpt).exists():
        cae=AnomalyCAE(ckpt=args.cae_ckpt)
    ml_det=None
    if args.use_ml and HAS_ML and Path(args.ml_ckpt).exists():
        ml_det=TrajectoryMLDetector(ckpt=args.ml_ckpt)
    elif args.use_ml:
        print(f"ML ckpt {args.ml_ckpt} not found. Train first: python scripts/train_ml_anomaly.py --source <normal_video>")

    cap=get_capture(args.source)
    is_images=isinstance(cap, list)
    if not is_images and not cap.isOpened():
        print(f"Cannot open {args.source}"); return
    # writer
    writer=None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=None  # init on first frame

    frame_idx=0; anomalies=0
    csv_path=Path(args.save).with_suffix(".csv") if args.save else Path("runs/infer/alerts.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file=open(csv_path,"w", newline=""); csv_writer=csv.writer(csv_file); csv_writer.writerow(["frame","track_id","reason","mse"])

    def process_frame(frame):
        nonlocal frame_idx, anomalies
        res=model.predict(frame, conf=args.conf, verbose=False)[0]
        dets=[]
        if res.boxes is not None and len(res.boxes)>0:
            for box in res.boxes:
                if int(box.cls)==0 or model.names[int(box.cls)]=="person":
                    x1,y1,x2,y2=box.xyxy[0].tolist(); conf=float(box.conf); cls=int(box.cls)
                    dets.append([x1,y1,x2,y2,conf,cls])
        dets=np.array(dets) if dets else np.zeros((0,6))
        tracks=tracker.update(dets, frame)
        frame_anom, per_track = detector_rule.update(tracks, tracker.analyzer, frame_idx)
        # own ML trajectory model fusion
        if ml_det is not None:
            for x1,y1,x2,y2,tid,conf in tracks:
                v=tracker.analyzer.velocity(tid); a=tracker.analyzer.direction(tid)
                flow=tracker.analyzer.flow_angle
                adiff=0.0
                if a is not None and flow is not None:
                    d=abs(a-flow); adiff=min(d,360-d)
                dwell=len(tracker.analyzer.hist.get(tid,[]))
                w,h=(x2-x1),(y2-y1)
                feat=extract_features(v,adiff,dwell,w,h,0.0)
                _, is_ml=ml_det.score(feat)
                if is_ml and tid not in per_track:
                    per_track[tid]=per_track.get(tid,"")+"ml-anomaly"
                    frame_anom=True
        # cae score
        mse=None; cae_anom=False
        if cae is not None:
            mse,cae_anom = cae.score(frame)
            frame_anom = frame_anom or cae_anom
        # draw
        for x1,y1,x2,y2,tid,conf in tracks:
            color=(0,255,0) if tid not in per_track else (0,0,255)
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            label=f"ID {tid}"
            if tid in per_track: label+=f" ANOM:{per_track[tid][:20]}"
            cv2.putText(frame,label,(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
            if tid in per_track:
                csv_writer.writerow([frame_idx,tid,per_track[tid],f"{mse:.5f}" if mse else ""])
        if frame_anom:
            cv2.putText(frame,"ANOMALY",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            anomalies+=1
        # stats
        cv2.putText(frame,f"Frame {frame_idx} Tracks {len(tracks)} Anomalies {anomalies}",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        frame_idx+=1
        return frame, frame_anom

    if is_images:
        for img_path in cap:
            frame=cv2.imread(str(img_path))
            out, _ = process_frame(frame)
            cv2.imshow("Anomaly", out)
            if writer:
                if writer is None: writer=cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),30,(out.shape[1],out.shape[0]))
                writer.write(out)
            if cv2.waitKey(1)==27: break
    else:
        while True:
            ret,frame=cap.read()
            if not ret: break
            if writer is None and args.save:
                h,w=frame.shape[:2]
                writer=cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),30,(w,h))
            out,_=process_frame(frame)
            cv2.imshow("Anomaly Detection (q to quit)", out)
            if writer: writer.write(out)
            if cv2.waitKey(1)==ord('q'): break
            # limit fps for 1650 display
            time.sleep(0.01)

    csv_file.close()
    if writer: writer.release()
    if not is_images: cap.release()
    cv2.destroyAllWindows()
    print(f"Done frames {frame_idx} anomalies {anomalies} csv {csv_path}")

if __name__=="__main__":
    main()
