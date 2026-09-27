"""
Modern tracker wrapper: ByteTrack (default) + BoT-SORT fallback via boxmot.
Works with Ultralytics YOLO output.
"""
import cv2
import numpy as np
from collections import deque, defaultdict

try:
    from boxmot import ByteTracker
    HAS_BOXMOT = True
except:
    HAS_BOXMOT = False

class TrajectoryAnalyzer:
    def __init__(self, max_len=30, max_tracks=80):
        self.hist = defaultdict(lambda: deque(maxlen=max_len))
        self.flow_angle = None
        self.max_tracks = max_tracks
        self.last_seen = {}  # track_id -> frame_idx

    def update(self, track_id, xyxy, frame_idx=None):
        x1,y1,x2,y2 = xyxy
        cx = (x1+x2)/2; cy = (y1+y2)/2
        self.hist[track_id].append((cx,cy))
        if frame_idx is not None:
            self.last_seen[track_id] = frame_idx
            # prune stale tracks to prevent memory leak / freeze
            if len(self.hist) > self.max_tracks:
                oldest = min(self.last_seen, key=self.last_seen.get)
                del self.hist[oldest]
                del self.last_seen[oldest]

    def prune(self, cur_frame, keep_frames=90):
        # remove tracks not seen for keep_frames (3 sec @30fps)
        to_del = [tid for tid, last in self.last_seen.items() if cur_frame - last > keep_frames]
        for tid in to_del:
            del self.hist[tid]
            del self.last_seen[tid]

    def velocity(self, track_id):
        h = self.hist[track_id]
        if len(h) < 5: return 0.0
        # distance over last 5 frames
        d = 0
        for i in range(1,5):
            x1,y1 = h[-i]; x2,y2 = h[-i-1]
            d += ((x1-x2)**2 + (y1-y2)**2)**0.5
        return d/5  # px/frame

    def direction(self, track_id):
        h = self.hist[track_id]
        if len(h) < 5: return None
        x1,y1 = h[-5]; x2,y2 = h[-1]
        import math
        return math.degrees(math.atan2(y2-y1, x2-x1))

    def calibrate_flow(self, angles):
        # dominant flow median
        if angles:
            self.flow_angle = float(np.median(angles))

class TrackerWrapper:
    def __init__(self, tracker_type="bytetrack", fps=30):
        self.tracker_type = tracker_type
        self.analyzer = TrajectoryAnalyzer()
        if HAS_BOXMOT and tracker_type=="bytetrack":
            # BoxMOT ByteTracker expects specific args
            from boxmot.trackers.bytetracker import ByteTracker as BT
            self.tracker = BT(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=fps)
        else:
            # Fallback simple IOU tracker (no deps)
            self.tracker = None
            self._next_id = 1
            self._tracks = {}  # id-> xyxy

    def update(self, dets, frame, frame_idx=0):
        """
        dets: numpy array Nx6 = x1,y1,x2,y2,conf,cls or empty
        frame: BGR image for ReID if needed
        returns: list of [x1,y1,x2,y2,id,conf]
        """
        # periodic prune to prevent freeze/memory leak
        if frame_idx % 30 == 0:
            self.analyzer.prune(frame_idx)
        if HAS_BOXMOT and self.tracker is not None:
            # boxmot expects (N,6) torch or numpy
            import torch
            if len(dets)==0:
                dets_tensor = np.zeros((0,6))
            else:
                dets_tensor = dets
            # boxmot update returns array with id in col 4?
            # Use boxmot API: update(dets, img)
            try:
                results = self.tracker.update(dets_tensor, frame)
                # results shape: N x (x1 y1 x2 y2 id conf cls)
                out=[]
                for r in results:
                    x1,y1,x2,y2,tid,conf,cls = r[:7] if len(r)>=7 else (*r[:4], r[4], r[5] if len(r)>5 else 0.9, 0)
                    out.append([float(x1),float(y1),float(x2),float(y2),int(tid),float(conf)])
                    self.analyzer.update(int(tid), (float(x1),float(y1),float(x2),float(y2)), frame_idx=frame_idx)
                return out
            except Exception as e:
                print(f"tracker error {e}, fallback")
        # fallback IOU tracker
        return self._iou_update(dets, frame_idx=frame_idx)

    def _iou_update(self, dets, iou_thresh=0.3, frame_idx=0):
        # very simple: greedy match
        if len(dets)==0:
            return []
        dets = sorted(dets, key=lambda x: x[4], reverse=True)
        new_tracks={}
        used=set()
        out=[]
        for det in dets:
            x1,y1,x2,y2,conf,cls = det
            best_id=None; best_iou=0
            for tid, txy in self._tracks.items():
                if tid in used: continue
                iou = self._iou((x1,y1,x2,y2), txy)
                if iou>best_iou and iou>iou_thresh:
                    best_iou=iou; best_id=tid
            if best_id is not None:
                new_tracks[best_id]=(x1,y1,x2,y2)
                used.add(best_id)
                out.append([x1,y1,x2,y2,best_id,conf])
                self.analyzer.update(best_id,(x1,y1,x2,y2), frame_idx=frame_idx)
            else:
                nid=self._next_id; self._next_id+=1
                new_tracks[nid]=(x1,y1,x2,y2)
                out.append([x1,y1,x2,y2,nid,conf])
                self.analyzer.update(nid,(x1,y1,x2,y2), frame_idx=frame_idx)
        self._tracks=new_tracks
        return out

    def _iou(self,a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        area_a=(a[2]-a[0])*(a[3]-a[1]); area_b=(b[2]-b[0])*(b[3]-b[1])
        return inter/(area_a+area_b-inter+1e-6)
