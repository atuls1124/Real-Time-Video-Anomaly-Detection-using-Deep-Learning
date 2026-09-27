"""
LIVE CCTV / WEBCAM ANOMALY DETECTION - One-click live demo
Works with: built-in webcam (0), external USB cam (1), IP CCTV RTSP, video file
Optimized for GTX 1650 4GB - ~20-30 FPS at 640px

Usage:
  python live.py                          # default webcam 0
  python live.py --source 0 --conf 0.35
  python live.py --source 1
  python live.py --source rtsp://admin:pass@192.168.1.64/stream
  python live.py --source video.mp4 --save runs/live/output.mp4
  python live.py --source 0 --weights models/yolo11s_mot17.pt
"""
import argparse, cv2, time, csv, json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from src.tracking.tracker import TrackerWrapper
from src.anomaly.rules import RuleAnomalyDetector
try:
    from src.anomaly.pose import PoseFilter
    HAS_POSE=True
except: HAS_POSE=False

def parse_args():
    p=argparse.ArgumentParser(description="Live anomaly detection")
    p.add_argument("--source", type=str, default="0", help="0/1 webcam, rtsp:// URL, or video path")
    p.add_argument("--weights", type=str, default="yolo11s.pt", help="yolo weights")
    p.add_argument("--conf", type=float, default=0.40, help="detection confidence (raise to 0.5 to reduce false positives)")
    p.add_argument("--save", type=str, default=None, help="save output mp4")
    p.add_argument("--no-show", action="store_true", help="headless, no window")
    p.add_argument("--sens", type=str, default="low", choices=["low","medium","high"], help="anomaly sensitivity")
    p.add_argument("--calm", action="store_true", help="calm/sitting mode - disables direction/loiter (use when testing sitting)")
    p.add_argument("--pose", action="store_true", help="enable pose filter to suppress sitting false positives (needs yolo11n-pose.pt)")
    p.add_argument("--roi", type=str, default=None, help="ROI x1,y1,x2,y2 normalized 0-1 e.g., 0.2,0.2,0.8,0.8 - only flag anomalies inside zone")
    p.add_argument("--fps-limit", type=int, default=0, help="cap FPS e.g., 20 to reduce GPU load and freeze")
    p.add_argument("--imgsz", type=int, default=384, help="inference size 320 fast / 384 balanced / 480 accurate")
    p.add_argument("--every", type=int, default=2, help="detect every Nth frame (2 = 2x faster, tracking interpolates)")
    return p.parse_args()

def open_source(src):
    # webcam index -> try OpenCV first, then FFmpeg fallback (Store Python blocks cv2)
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            try:
                from src.utils.ffmpeg_cam import FFmpegCamera
                print("[CAM] OpenCV blocked, using FFmpeg fallback")
                cap = FFmpegCamera()
            except Exception as e:
                print(f"[CAM] FFmpeg fallback failed {e}")
        else:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    else:
        cap = cv2.VideoCapture(src)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        return cap

def main():
    args=parse_args()
    w = Path(args.weights)
    if not w.exists() and str(w).startswith("models"):
        print(f"[WARN] {w} not found, using pretrained yolo11s.pt")
        w = Path("yolo11s.pt")
    print(f"[LOAD] {w} conf={args.conf} source={args.source}")
    model = YOLO(str(w))
    # warmup
    dummy = np.zeros((640,640,3), dtype=np.uint8)
    model.predict(dummy, verbose=False)

    # sensitivity presets - loiter DISABLED by default for webcam (would flag sitting). Use --enable-loiter if you need it for surveillance
    sens_map = {
        "low":    dict(angle_thresh=130, dwell_thresh=999999, min_vel_for_angle=11, persist_frames=5, vel_floor=16),
        "medium": dict(angle_thresh=115, dwell_thresh=999999, min_vel_for_angle=9,  persist_frames=4, vel_floor=14),
        "high":   dict(angle_thresh=90,  dwell_thresh=999999, min_vel_for_angle=7,  persist_frames=3, vel_floor=12),
    }
    cfg = sens_map[args.sens]
    if args.calm:
        cfg["angle_thresh"]=180  # disable direction
        print("[MODE] calm/sitting - direction & loiter disabled, only fast motion flagged")
    # loiter is now disabled for live webcam - sitting will NOT be flagged

    tracker = TrackerWrapper(tracker_type="bytetrack")
    detector = RuleAnomalyDetector(auto_calib_frames=45, angle_thresh=cfg["angle_thresh"],
                                   dwell_thresh=cfg["dwell_thresh"], min_vel_for_angle=cfg["min_vel_for_angle"],
                                   persist_frames=cfg["persist_frames"], cooldown_frames=45)
    # force floor via vel_thresh preset if needed
    if cfg["vel_floor"] > 14:
        detector.vel_thresh = float(cfg["vel_floor"])
    # ROI
    if args.roi:
        try:
            roi = tuple(map(float, args.roi.split(",")))
            assert len(roi)==4
            detector.set_roi(roi)
        except: print(f"[WARN] bad ROI {args.roi}, use x1,y1,x2,y2 0-1")
    # pose filter
    pose_filter=None
    if args.pose and HAS_POSE:
        pose_filter=PoseFilter()
        if not pose_filter.enabled:
            pose_filter=None
            print("[Pose] fallback disabled")
        else:
            print("[Pose] sitting suppression enabled")

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source {args.source}")
        print("Try: --source 1  or --source rtsp://...  or check camera permission")
        return
    # threaded non-blocking capture fixes stuck webcam
    try:
        from src.utils.ffmpeg_cam import ThreadedCamera
        cap = ThreadedCamera(cap)
        print("[CAM] threaded capture ON (no-stuck)")
    except Exception as e:
        print(f"[CAM] threaded wrapper skipped {e}")

    # get props
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"[CAM] {W}x{H} @ {fps:.1f} FPS - Press 'q' to quit, 's' to save snapshot")

    writer=None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
        print(f"[SAVE] -> {args.save}")

    csv_path = Path("runs/live/alerts.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame","track_id","reason"])

    frame_idx=0
    anomalies=0
    start=time.time()
    last_fps_update=start
    fps_show=0
    detect_every = max(1, args.every)  # 2 = infer every 2nd frame, 2x faster on 1650
    last_dets = np.zeros((0,6))
    loop_times = []
    print(f"[PERF] imgsz={args.imgsz} every={detect_every} (increase --every 3 if still stuck)")

    cv2.namedWindow("LIVE Anomaly - CCTV/Webcam [q=quit s=snapshot]", cv2.WINDOW_NORMAL)
    if W>1280:
        cv2.resizeWindow("LIVE Anomaly - CCTV/Webcam [q=quit s=snapshot]", 1280, 720)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue
        # inference with throttling to prevent freeze on 4GB VRAM
        if frame_idx % detect_every == 0:
            res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            dets=[]
            if res.boxes is not None and len(res.boxes)>0:
                for box in res.boxes:
                    cls=int(box.cls)
                    name=model.names[cls]
                    if name=="person" or cls==0:
                        x1,y1,x2,y2=box.xyxy[0].tolist()
                        conf=float(box.conf)
                        dets.append([x1,y1,x2,y2,conf,cls])
            last_dets = np.array(dets) if dets else np.zeros((0,6))
        dets = last_dets
        tracks = tracker.update(dets, frame, frame_idx=frame_idx)
        # pose sitting suppression
        if pose_filter and tracks:
            sitting_ids = pose_filter.filter_tracks(frame, tracks)
            # force those tracks to not be flagged by clearing their velocity history? just remove from per_track after
            # we let detector run then filter
        is_anom, per_track = detector.update(tracks, tracker.analyzer, frame_idx, frame_shape=frame.shape)
        if pose_filter and per_track:
            # remove sitting persons from anomaly
            per_track = {tid: reason for tid, reason in per_track.items() if tid not in pose_filter.filter_tracks(frame, tracks)}
            is_anom = len(per_track)>0
        # periodic GPU cleanup to prevent freeze after ~5 mins
        if frame_idx % 120 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # FPS limit to prevent freeze on 1650
        if args.fps_limit and args.fps_limit>0:
            time.sleep(max(0, 1.0/args.fps_limit - 0.02))

        # draw
        for x1,y1,x2,y2,tid,conf in tracks:
            is_bad = tid in per_track
            color = (0,0,255) if is_bad else (0,255,0)
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            label=f"ID {tid} {conf:.2f}"
            if is_bad:
                label+=f" !{per_track[tid][:22]}"
            cv2.putText(frame, label, (int(x1), max(15,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if is_bad:
                csv_writer.writerow([frame_idx, tid, per_track[tid]])
        if is_anom:
            anomalies+=1
            cv2.rectangle(frame, (0,0), (W,40), (0,0,255), -1)
            cv2.putText(frame, f"ANOMALY DETECTED! {len(per_track)} persons", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            # optional beep: print bell
            print(f"\a[ANOMALY] frame {frame_idx} ids {list(per_track.keys())} {list(per_track.values())}")

        # calibration / status bar
        if not detector.calibrated:
            cv2.putText(frame, f"Calibrating... {frame_idx}/{detector.auto_calib_frames} (walk normally)", (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        else:
            thr = detector.vel_thresh if detector.vel_thresh else 0
            cv2.putText(frame, f"vel_thr {thr:.1f} flow {detector.flow_angle:.0f}deg" if detector.flow_angle else f"vel_thr {thr:.1f}", (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # FPS & stats
        now=time.time()
        if now - last_fps_update >= 1.0:
            fps_show = frame_idx / (now - start) if frame_idx else 0
            last_fps_update=now
        cv2.rectangle(frame, (0, H-28), (W, H), (0,0,0), -1)
        cv2.putText(frame, f"FPS {fps_show:.1f} | Frame {frame_idx} | Tracks {len(tracks)} | Anomalies {anomalies} | q=quit s=snap", (8, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

        if not args.no_show:
            # resize large frames for display to avoid imshow stuck
            disp = frame
            if disp.shape[1] > 960:
                disp = cv2.resize(disp, (960, int(960*disp.shape[0]/disp.shape[1])))
            cv2.imshow("LIVE Anomaly - CCTV/Webcam [q=quit s=snapshot]", disp)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break
            if key==ord('s'):
                snap=f"runs/live/snap_{frame_idx}.jpg"
                cv2.imwrite(snap, frame)
                print(f"[SNAP] {snap}")

        if writer:
            writer.write(frame)

        frame_idx+=1
        # heartbeat so console never looks stuck (even when calm = no anomaly)
        if frame_idx % 60 == 0:
            avg_ms = (sum(loop_times[-60:])/60*1000) if len(loop_times)>=60 else 0
            print(f"[LIVE] frame {frame_idx} FPS {fps_show:.1f} loop {avg_ms:.0f}ms tracks {len(tracks)} anomalies {anomalies} (sitting calm = silent is normal)")
        # watchdog: track loop time, auto-degrade if stuck
        loop_times.append(time.time()-t0)
        if len(loop_times) > 60:
            avg = sum(loop_times[-60:])/60
            if avg > 0.12 and detect_every < 3:
                detect_every += 1
                print(f"[PERF] slow loop {avg*1000:.0f}ms, auto every->{detect_every}")
                loop_times.clear()
        # auto-reset after 10 mins to prevent indefinite memory growth
        if frame_idx > 18000:
            print("[RESET] 10 min reached, resetting tracker to prevent freeze")
            tracker = TrackerWrapper(tracker_type="bytetrack")
            detector = RuleAnomalyDetector(auto_calib_frames=45)
            frame_idx=0

    csv_file.close()
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"[DONE] frames {frame_idx} anomalies {anomalies} csv {csv_path}")
    print(f"Calibration: vel_thresh={detector.vel_thresh} flow={detector.flow_angle}")

if __name__=="__main__":
    main()
