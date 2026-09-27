"""
Rule-based anomaly detector v2: ROI zones, cooldown, adaptive threshold, sitting filter.
Handles webcam sitting vs CCTV surveillance correctly.
"""
import math, time
import numpy as np
from collections import defaultdict, deque

class RuleAnomalyDetector:
    def __init__(self, vel_thresh=None, angle_thresh=115, dwell_thresh=999999, auto_calib_frames=60,
                 min_vel_for_angle=9.0, persist_frames=4, cooldown_frames=30, roi=None):
        self.vel_thresh = vel_thresh
        self.angle_thresh = angle_thresh
        self.dwell_thresh = dwell_thresh  # disabled by default for webcam
        self.auto_calib_frames = auto_calib_frames
        self.min_vel_for_angle = min_vel_for_angle
        self.persist_frames = persist_frames
        self.cooldown_frames = cooldown_frames  # prevent spam
        self.roi = roi  # [(x1,y1),(x2,y2)] or None = whole frame
        self.vel_history = []
        self.angle_history = []
        self.calibrated = False
        self.flow_angle = None
        self.frame_count = 0
        self.track_birth = {}
        self.anomaly_log = []
        self._consecutive = {}
        self._last_alert = {}  # tid -> frame_idx
        self._ema_vel = None

    def set_roi(self, roi):
        self.roi = roi  # None or (x1,y1,x2,y2) normalized 0-1 or pixels
        print(f"[Anomaly] ROI set {roi}")

    def _in_roi(self, xyxy, frame_shape=None):
        if self.roi is None:
            return True
        x1,y1,x2,y2 = xyxy
        cx = (x1+x2)/2; cy = (y1+y2)/2
        # roi can be pixels or normalized
        rx1,ry1,rx2,ry2 = self.roi
        if max(self.roi) <= 1.0 and frame_shape:
            h,w = frame_shape[:2]
            rx1*=w; rx2*=w; ry1*=h; ry2*=h
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def calibrate(self, velocities, angles):
        if self.vel_thresh is None and len(velocities)>40:
            mean = np.mean(velocities); std = np.std(velocities)
            raw = float(mean + 2.8*std)
            self.vel_thresh = max(raw, 14.0)
            # EMA init
            self._ema_vel = mean
            print(f"[Anomaly] calibrated vel_thresh={self.vel_thresh:.2f} (mean {mean:.2f} std {std:.2f} floor 14)")
        if self.flow_angle is None and len(angles)>15:
            if len(self.vel_history)>0 and np.mean(self.vel_history) > 6:
                self.flow_angle = float(np.median(angles))
                print(f"[Anomaly] calibrated flow_angle={self.flow_angle:.1f}")
            else:
                print(f"[Anomaly] sitting scene - direction disabled")

    def update(self, tracks, analyzer, frame_idx, frame_shape=None):
        self.frame_count=frame_idx
        per_track={}
        velocities=[]
        angles=[]
        for x1,y1,x2,y2,tid,conf in tracks:
            v = analyzer.velocity(tid)
            a = analyzer.direction(tid)
            if v>0: velocities.append(v)
            if a is not None and v > self.min_vel_for_angle: angles.append(a)
            if tid not in self.track_birth: self.track_birth[tid]=frame_idx

        # adaptive EMA threshold after calibration
        if self.calibrated and velocities:
            cur_mean = np.mean(velocities)
            if self._ema_vel is None: self._ema_vel = cur_mean
            else: self._ema_vel = 0.95*self._ema_vel + 0.05*cur_mean
            # slowly adapt thresh (only upwards to avoid drift to sitting)
            if cur_mean < self._ema_vel*0.5:
                pass  # sitting, don't lower thresh
            elif self.vel_thresh and cur_mean > self.vel_thresh*0.8:
                # scene getting faster, nudge thresh
                self.vel_thresh = 0.99*self.vel_thresh + 0.01*(cur_mean+12)

        if not self.calibrated and frame_idx < self.auto_calib_frames:
            self.vel_history.extend([v for v in velocities if v>1])
            self.angle_history.extend(angles)
            if frame_idx==self.auto_calib_frames-1:
                self.calibrate(self.vel_history, self.angle_history)
                self.calibrated=True

        for x1,y1,x2,y2,tid,conf in tracks:
            # ROI filter - ignore persons outside zone (e.g., sitting at desk outside surveillance zone)
            if not self._in_roi((x1,y1,x2,y2), frame_shape):
                self._consecutive[tid]=0
                continue
            v = analyzer.velocity(tid)
            a = analyzer.direction(tid)
            dwell = frame_idx - self.track_birth.get(tid, frame_idx)
            # cooldown check
            if tid in self._last_alert and frame_idx - self._last_alert[tid] < self.cooldown_frames:
                continue
            reasons=[]
            if self.calibrated and self.vel_thresh and v > self.vel_thresh:
                # extra check: sitting jitter has high vel but tiny bbox movement? require bbox size stable
                reasons.append(f"speed {v:.1f}>{self.vel_thresh:.1f}")
            if self.calibrated and self.flow_angle is not None and a is not None and v > self.min_vel_for_angle:
                diff = abs(a - self.flow_angle)
                diff = min(diff, 360-diff)
                if diff > self.angle_thresh:
                    reasons.append(f"dir {diff:.0f}°")
            # loiter only if ROI enabled and v low
            if self.roi is not None and dwell > self.dwell_thresh and v < 4:
                reasons.append(f"loiter {dwell}f")
            # pose helper: if v very low (<2) never direction/loiter even if dwell large
            if v < 2.0:
                reasons = [r for r in reasons if not r.startswith("dir") and not r.startswith("loiter")]

            if reasons:
                cnt = self._consecutive.get(tid, 0) + 1
                self._consecutive[tid]=cnt
                if cnt >= self.persist_frames:
                    per_track[tid]=",".join(reasons)
                    self.anomaly_log.append({"frame":frame_idx,"id":tid,"reason":per_track[tid],"xyxy":(x1,y1,x2,y2)})
                    self._last_alert[tid]=frame_idx
            else:
                self._consecutive[tid]=0

        return len(per_track)>0, per_track

    def reset(self):
        self.vel_history.clear(); self.angle_history.clear(); self.calibrated=False
        self._consecutive.clear(); self._last_alert.clear()
