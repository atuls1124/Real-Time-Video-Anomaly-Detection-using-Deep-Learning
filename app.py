import streamlit as st
import cv2, tempfile, csv, time, json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from ultralytics import YOLO
from src.tracking.tracker import TrackerWrapper
from src.anomaly.rules import RuleAnomalyDetector
try:
    from src.anomaly.pose import PoseFilter
    HAS_POSE=True
except: HAS_POSE=False

st.set_page_config(page_title="Anomaly Detection Surveillance", layout="wide", page_icon="🚨")

st.markdown("""
<style>
.main {background: #0e1117}
h1 {color:#00d4ff}
.stAlert {border-radius: 10px}
</style>
""", unsafe_allow_html=True)

st.title("🚨 Anomaly Detection in Surveillance Videos - Real Time")
st.caption("YOLO11s + ByteTrack + Rule/CAE/Pose | GTX 1650 Optimized | Auto-calibrated + ROI")

with st.sidebar:
    st.header("⚙️ Settings")
    weights = st.selectbox("Detector Weights", ["yolo11s.pt","yolo11n.pt","models/yolo11s_mot17.pt"], index=0)
    conf = st.slider("Confidence", 0.1,0.8,0.40,0.05)
    tracker_type = st.selectbox("Tracker", ["bytetrack","iou (fallback)"])
    sens = st.select_slider("Sensitivity", options=["low","medium","high"], value="low")
    calm = st.checkbox("Calm / Sitting mode (no loiter/dir)", value=True, help="Enable when testing sitting indoors - eliminates false loiter")
    use_pose = st.checkbox("Pose filter (sitting suppression)", value=False)
    use_roi = st.checkbox("Enable ROI zone", value=False)
    if use_roi:
        roi_x = st.slider("ROI X", 0.0,1.0,(0.0,1.0))
        roi_y = st.slider("ROI Y", 0.0,1.0,(0.0,1.0))
        st.caption("ROI = normalized x1,y1,x2,y2 - anomaly only inside zone")
    else:
        roi_x, roi_y = (0.0,1.0), (0.0,1.0)
    st.divider()
    st.info("For live CCTV use `python live.py --source 0 --calm` for best stability.")
    st.markdown("**Rules:** Speed > floor 14 + 2.8σ, Dir >115° if moving >9px/f, persist 4-5 frames, cooldown 45f")

tab1, tab2 = st.tabs(["📹 Live Inference", "📊 Metrics"])

with tab1:
    source = st.radio("Source", ["Upload Video","Webcam (0)","IP CCTV (RTSP)","Sample MOT17"], horizontal=True)
    uploaded=None
    rtsp_url=""
    if source=="Upload Video":
        uploaded=st.file_uploader("Upload surveillance clip", type=["mp4","avi","mov","mkv"])
    if source=="IP CCTV (RTSP)":
        rtsp_url=st.text_input("RTSP URL", "rtsp://admin:password@192.168.1.64:554/stream")
    col1,col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start Detection", type="primary")
    with col2:
        stop_btn = st.button("⏹ Stop")
    if start_btn:
        if source=="Upload Video" and uploaded:
            tfile=tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            src=tfile.name
        elif source=="Webcam (0)":
            src=0
        elif source=="IP CCTV (RTSP)":
            if not rtsp_url: st.error("Enter RTSP URL"); st.stop()
            src=rtsp_url
        else:
            sample = Path("data/MOT17-YOLO/images/val")
            if sample.exists() and list(sample.glob("*.jpg")):
                src = str(list(sample.glob("*.jpg"))[0].parent)
            else:
                st.error("No sample found, upload video"); st.stop()

        with st.spinner("Loading YOLO11..."):
            if not Path(weights).exists() and weights.startswith("models"):
                st.warning(f"{weights} not found, using yolo11s.pt")
                weights_use="yolo11s.pt"
            else: weights_use=weights
            model=YOLO(weights_use)
            tracker=TrackerWrapper(tracker_type="bytetrack" if tracker_type=="bytetrack" else "iou")
            sens_map={"low":dict(angle_thresh=130,dwell_thresh=999999,min_vel_for_angle=11,persist_frames=5,vel_floor=16),
                      "medium":dict(angle_thresh=115,dwell_thresh=999999,min_vel_for_angle=9,persist_frames=4,vel_floor=14),
                      "high":dict(angle_thresh=90,dwell_thresh=999999,min_vel_for_angle=7,persist_frames=3,vel_floor=12)}
            cfg=sens_map[sens]
            if calm: cfg["angle_thresh"]=180
            detector=RuleAnomalyDetector(auto_calib_frames=45, angle_thresh=cfg["angle_thresh"], dwell_thresh=cfg["dwell_thresh"],
                                         min_vel_for_angle=cfg["min_vel_for_angle"], persist_frames=cfg["persist_frames"], cooldown_frames=45)
            if cfg["vel_floor"]>14: detector.vel_thresh=float(cfg["vel_floor"])
            if use_roi:
                detector.set_roi((roi_x[0], roi_y[0], roi_x[1], roi_y[1]))
            pose_filter=None
            if use_pose and HAS_POSE:
                pose_filter=PoseFilter()
                if not pose_filter.enabled: pose_filter=None
        st.success(f"Loaded {weights_use} | sens {sens} | calm={calm} | pose={pose_filter is not None}")

        # open capture with FFmpeg fallback (Store Python blocks cv2.VideoCapture)
        def open_cap(s):
            if isinstance(s, str) and Path(s).is_dir():
                return None
            c = cv2.VideoCapture(s)
            if c.isOpened():
                try: c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except: pass
                st.caption("Camera: OpenCV")
                return c
            try:
                from src.utils.ffmpeg_cam import FFmpegCamera
                f = FFmpegCamera()
                if f.isOpened():
                    st.caption("Camera: FFmpeg fallback (OpenCV blocked)")
                    return f
            except Exception as e:
                st.error(f"FFmpeg fallback failed {e}")
            return c  # will fail isOpened check below
        cap = open_cap(src)
        if cap is not None and not cap.isOpened():
            st.error("Webcam not opening. Fix: 1) Close Teams/Zoom/Camera app 2) Settings > Privacy > Camera > Allow desktop apps 3) Use Upload Video instead (always works). FFmpeg fallback also failed - check ffmpeg installed.")
            st.stop()
        is_img_dir = isinstance(src,str) and Path(src).is_dir()
        placeholder=st.empty()
        chart_placeholder=st.empty()
        alert_placeholder=st.empty()
        status_placeholder=st.empty()
        frame_idx=0; anomalies_frames=[]
        last_sitting_warn=False

        if is_img_dir:
            imgs=sorted(Path(src).glob("*.jpg"))[:150]
            for imgp in imgs:
                frame=cv2.imread(str(imgp))
                res=model.predict(frame, conf=conf, verbose=False, imgsz=480)[0]
                dets=[]
                if res.boxes is not None:
                    for b in res.boxes:
                        x1,y1,x2,y2=b.xyxy[0].tolist(); cf=float(b.conf)
                        dets.append([x1,y1,x2,y2,cf,0])
                dets=np.array(dets) if dets else np.zeros((0,6))
                tracks=tracker.update(dets, frame, frame_idx=frame_idx)
                # pose filter
                sitting=set()
                if pose_filter and tracks:
                    sitting=pose_filter.filter_tracks(frame, tracks)
                is_anom, per = detector.update(tracks, tracker.analyzer, frame_idx, frame_shape=frame.shape)
                if sitting: per={k:v for k,v in per.items() if k not in sitting}
                for x1,y1,x2,y2,tid,cf in tracks:
                    is_sit = tid in sitting
                    is_bad = tid in per
                    col=(255,180,0) if is_sit else (0,0,255) if is_bad else (0,255,0)
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),col,2)
                    label=f"ID{tid}"+(" SIT" if is_sit else " ANOM" if is_bad else "")
                    cv2.putText(frame,label,(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
                if is_anom and per: cv2.putText(frame,"ANOMALY",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                if sitting and not last_sitting_warn:
                    status_placeholder.info(f"🧍 Pose: {len(sitting)} sitting suppressed")
                placeholder.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), width='stretch')
                anomalies_frames.append(1 if per else 0)
                frame_idx+=1
                time.sleep(0.05)
        else:
            import time as _tm
            _t0=_tm.time(); _shown=0; tracks=[]; per={}
            status_placeholder.info("Starting webcam... first frames slow (CUDA warmup ~10s), then live.")
            while cap.isOpened():
                ret,frame=cap.read()
                if not ret or frame is None:
                    time.sleep(0.02); continue
                # frame skip: infer every 3rd, show every 2nd -> browser never freezes
                do_infer = (frame_idx % 3 == 0)
                if do_infer:
                    # downscale for speed
                    h0,w0=frame.shape[:2]
                    if w0>640:
                        frame_small=cv2.resize(frame,(640,int(640*h0/w0)))
                    else: frame_small=frame
                    res=model.predict(frame_small, conf=conf, verbose=False, imgsz=384)[0]
                    dets=[]
                    if res.boxes is not None:
                        sx=w0/frame_small.shape[1]; sy=h0/frame_small.shape[0]
                        for b in res.boxes:
                            x1,y1,x2,y2=b.xyxy[0].tolist(); cf=float(b.conf)
                            dets.append([x1*sx,y1*sy,x2*sx,y2*sy,cf,0])
                    dets=np.array(dets) if dets else np.zeros((0,6))
                    tracks=tracker.update(dets, frame, frame_idx=frame_idx)
                # show every 2nd frame to keep browser responsive
                if frame_idx % 2 == 0:
                    pass  # fall through to draw+show below
                elif not do_infer:
                    frame_idx+=1; continue
                sitting=set()
                if pose_filter and tracks and frame_idx % 3==0:
                    sitting=pose_filter.filter_tracks(frame, tracks)
                is_anom, per = detector.update(tracks, tracker.analyzer, frame_idx, frame_shape=frame.shape)
                if sitting: per={k:v for k,v in per.items() if k not in sitting}
                # draw ROI
                if use_roi:
                    h,w=frame.shape[:2]
                    cv2.rectangle(frame,(int(roi_x[0]*w),int(roi_y[0]*h)),(int(roi_x[1]*w),int(roi_y[1]*h)),(255,255,0),1)
                for x1,y1,x2,y2,tid,cf in tracks:
                    is_sit=tid in sitting
                    is_bad=tid in per
                    col=(255,180,0) if is_sit else (0,0,255) if is_bad else (0,255,0)
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),col,2)
                if per: cv2.putText(frame,f"ANOMALY {len(per)}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                # downscale display for browser speed
                disp=frame
                if disp.shape[1]>640:
                    disp=cv2.resize(disp,(640,int(640*disp.shape[0]/disp.shape[1])))
                placeholder.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB), width='stretch')
                anomalies_frames.append(1 if per else 0)
                frame_idx+=1
                if frame_idx % 30 == 0:
                    el=_tm.time()-_t0
                    status_placeholder.info(f"LIVE frame {frame_idx} | {frame_idx/max(el,1):.1f} FPS | tracks {len(tracks)} | anomalies {sum(anomalies_frames)}")
                if frame_idx>600: break

        fig=go.Figure()
        fig.add_trace(go.Scatter(y=anomalies_frames, mode="lines", name="Anomaly", line=dict(color="red")))
        fig.update_layout(title="Anomaly Timeline (1=anomaly, debounced + cooldown)", height=220, margin=dict(l=10,r=10,t=30,b=10))
        chart_placeholder.plotly_chart(fig, width='stretch')
        if sum(anomalies_frames)>0:
            alert_placeholder.error(f"🚨 {sum(anomalies_frames)} anomalous frames / {frame_idx} (pose suppressed sitting, loiter disabled)")
        else:
            alert_placeholder.success("✅ No anomalies - calm behavior correctly ignored")

with tab2:
    st.subheader("System Architecture v2")
    st.markdown("""
    **Pipeline:** `Video → YOLO11s (480px, person) → ByteTrack (Kalman) → Trajectory Analyzer (30f, prune) → Rule v2 (vel floor 14, dir only if >9px/f, persist 4-5, cooldown 45) + Pose sitting filter → ROI gate`  
    **Fixes:** Loiter disabled for webcam (999999), direction disabled in calm (180°), EMA adaptive, no freeze (buffersize 1, 480px, empty_cache)  
    **Datasets:** MOT17 4791/525 (mAP50 0.919) + Avenue placeholder for CAE  
    """)
    st.image("https://docs.ultralytics.com/images/yolo11-compare.png", caption="YOLO11 family")
    st.divider()
    st.json({"weights": weights, "conf": conf, "sens": sens, "calm": calm, "pose": use_pose, "roi": (roi_x, roi_y) if use_roi else None, "mot17_val": len(list(Path("data/MOT17-YOLO/images/val").glob("*.jpg"))) if Path("data/MOT17-YOLO/images/val").exists() else 0})
