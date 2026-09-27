"""
Own trajectory anomaly model: learns NORMAL motion from your CCTV.
Lightweight for GTX 1650 4GB: sklearn IsolationForest on 6-D features, CPU only.

Features per track per frame (from TrajectoryAnalyzer):
  [velocity px/f, angle-diff vs flow, dwell frames, bbox aspect w/h, area, area-change]

Train on NORMAL video only -> save models/ml_anomaly.pkl
Infer: score() -> anomaly if IsolationForest says -1, fused with Rule detector.

Usage:
  python scripts/train_ml_anomaly.py --source data/normal_clip.mp4 --epochs n/a
  python inference.py --source video.mp4 --use-ml --ml-ckpt models/ml_anomaly.pkl
"""
import pickle
import numpy as np
from pathlib import Path

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def extract_features(velocity, angle_diff, dwell, w, h, area_change):
    """6-D feature vector. angle_diff may be None -> 0."""
    if angle_diff is None:
        angle_diff = 0.0
    aspect = float(w / max(h, 1))
    return [float(velocity), float(angle_diff), float(dwell),
            float(aspect), float(w * h), float(area_change)]


class TrajectoryMLDetector:
    def __init__(self, ckpt=None, contamination=0.05):
        self.ckpt = ckpt
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.clf = IsolationForest(
            n_estimators=100, contamination=contamination,
            random_state=42, n_jobs=1) if HAS_SKLEARN else None
        self.trained = False
        if ckpt and Path(ckpt).exists():
            self.load(ckpt)

    def train(self, X_normal, save_path="models/ml_anomaly.pkl"):
        assert HAS_SKLEARN, "pip install scikit-learn"
        X = np.array(X_normal, dtype=np.float32)
        # drop degenerate rows (no motion history yet)
        X = X[X[:, 0] >= 0]
        if len(X) < 50:
            raise ValueError(f"Not enough normal samples ({len(X)}). Need >=50 frames of tracks.")
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs)
        self.trained = True
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "clf": self.clf}, f)
        print(f"[ML] trained on {len(X)} samples -> {save_path}")
        return save_path

    def load(self, ckpt):
        with open(ckpt, "rb") as f:
            d = pickle.load(f)
        self.scaler, self.clf = d["scaler"], d["clf"]
        self.trained = True
        print(f"[ML] loaded {ckpt}")

    def score(self, feat):
        """feat: 6-D list. Returns (anomaly_score, is_anom). Higher score = more abnormal."""
        if not self.trained:
            return 0.0, False
        Xs = self.scaler.transform([feat])
        pred = self.clf.predict(Xs)[0]  # 1 normal, -1 anomaly
        raw = -self.clf.score_samples(Xs)[0]  # higher = more abnormal
        return float(raw), bool(pred == -1)
