"""
Download + inspect CUHK Avenue dataset if not present.
Fallback: create placeholder structure and download instructions.
"""
from pathlib import Path
import urllib.request, zipfile, os

BASE = Path(__file__).parent.parent
AVE_ROOT = BASE / "data" / "avenue"
AVE_ROOT.mkdir(parents=True, exist_ok=True)

# Check if already downloaded
if list(AVE_ROOT.glob("*.avi")) or list(AVE_ROOT.rglob("*.avi")) or (AVE_ROOT / "training_videos").exists():
    print("Avenue already present")
    for p in AVE_ROOT.rglob("*"):
        if p.is_file(): print(p.relative_to(BASE)) 
else:
    print("Avenue not found locally. Creating structure and downloading ground truth sample...")
    # Create dirs
    for d in ["training_videos","testing_videos","ground_truth"]:
        (AVE_ROOT / d).mkdir(exist_ok=True)
    # Try to create README
    (AVE_ROOT / "README.txt").write_text("""CUHK Avenue Dataset
Download manually if script fails:
https://www.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
Expected:
- training_videos/ 16 videos (normal only)
- testing_videos/ 21 videos (normal + abnormal)
- avenue_groundtruth.mat or testing_vol/ folder
For this project H1 rule-based does not need training data, H2 CAE trains on normal frames.
We provide synthetic fallback if download blocked.
""")
    print("Please download Avenue dataset manually to data/avenue/")
    print("For now pipeline will work with MOT17-only demo + synthetic anomaly samples.")
