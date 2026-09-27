"""
Pose helper: uses YOLO11-pose to distinguish sitting vs standing/running.
Falls back gracefully if pose model not available.
Sitting = low hip-knee angle + low movement -> force NOT anomaly even if bbox jitters.
"""
import cv2, numpy as np
try:
    from ultralytics import YOLO
    HAS_YOLO=True
except: HAS_YOLO=False

class PoseFilter:
    def __init__(self, weights="yolo11n-pose.pt", conf=0.5):
        self.enabled=False
        self.model=None
        if HAS_YOLO:
            try:
                self.model=YOLO(weights)
                self.enabled=True
                print(f"[Pose] loaded {weights}")
            except Exception as e:
                print(f"[Pose] disabled {e}")
                self.enabled=False
        self.conf=conf

    def is_sitting(self, frame, xyxy):
        if not self.enabled: return False
        x1,y1,x2,y2=map(int, xyxy)
        # crop with padding
        pad=10
        h,w=frame.shape[:2]
        x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w,x2+pad); y2=min(h,y2+pad)
        crop=frame[y1:y2, x1:x2]
        if crop.size==0 or crop.shape[0]<40: return False
        try:
            res=self.model.predict(crop, verbose=False, conf=self.conf)[0]
            if res.keypoints is None or len(res.keypoints)==0: return False
            kpts=res.keypoints.xy[0].cpu().numpy()  # 17x2
            confs=res.keypoints.conf[0].cpu().numpy() if hasattr(res.keypoints,'conf') else np.ones(17)
            if len(kpts)<17: return False
            # COCO: 11 hip left, 12 hip right, 13 knee left, 14 knee right, 15 ankle left, 16 ankle right, 5 shoulder
            # check visibility
            if confs[11]<0.3 or confs[13]<0.3: return False
            # hip-knee vertical distance small when sitting (knees near hips vertically)
            hip_y=(kpts[11][1]+kpts[12][1])/2
            knee_y=(kpts[13][1]+kpts[15][1])/2 if confs[15]>0.3 else kpts[13][1]
            # sitting: hip and knee y close (< 0.3 * bbox height), standing: knee well below hip
            bbox_h=y2-y1
            if abs(knee_y - hip_y) < bbox_h*0.25:
                return True
            # also check knee angle: sitting ~ 90deg, standing ~180
            # approximate using hip-knee-ankle
            import math
            hip=np.array(kpts[11]); knee=np.array(kpts[13]); ankle=np.array(kpts[15])
            v1=hip-knee; v2=ankle-knee
            if np.linalg.norm(v1)<5 or np.linalg.norm(v2)<5: return False
            ang = math.degrees(math.acos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1,1)))
            if 70 < ang < 120:
                return True
            return False
        except Exception as e:
            return False

    def filter_tracks(self, frame, tracks):
        # returns set of tids that are sitting (should suppress anomaly)
        sitting=set()
        for x1,y1,x2,y2,tid,conf in tracks:
            if self.is_sitting(frame, (x1,y1,x2,y2)):
                sitting.add(tid)
        return sitting
