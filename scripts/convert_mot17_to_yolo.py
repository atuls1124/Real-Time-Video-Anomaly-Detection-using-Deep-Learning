"""
Convert MOT17 (mot/MOT17) to YOLO format for Ultralytics training.
Uses FRCNN variant only to avoid duplicate frames.
Train: 02,04,05,10,11,13  Val: 09  (09 is complex, good for val)
Handles 4GB VRAM by producing 640-safe dataset.
"""
import os, shutil
from pathlib import Path
import configparser

BASE = Path(__file__).parent.parent
MOT_ROOT = BASE / "mot" / "MOT17" / "train"
OUT_ROOT = BASE / "data" / "MOT17-YOLO"
IMG_TRAIN = OUT_ROOT / "images" / "train"
IMG_VAL = OUT_ROOT / "images" / "val"
LBL_TRAIN = OUT_ROOT / "labels" / "train"
LBL_VAL = OUT_ROOT / "labels" / "val"

VAL_SEQS = {"MOT17-09-FRCNN"}  # validation
TRAIN_SEQS = {"MOT17-02-FRCNN","MOT17-04-FRCNN","MOT17-05-FRCNN","MOT17-10-FRCNN","MOT17-11-FRCNN","MOT17-13-FRCNN"}

def ensure_dirs():
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    # clean previous
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        for f in d.glob("*"):
            if f.is_file(): f.unlink()

def convert_seq(seq_name):
    seq_path = MOT_ROOT / seq_name
    ini = configparser.ConfigParser()
    ini.read(seq_path / "seqinfo.ini")
    w = int(ini["Sequence"]["imWidth"])
    h = int(ini["Sequence"]["imHeight"])
    is_val = seq_name in VAL_SEQS
    img_src = seq_path / "img1"
    gt_file = seq_path / "gt" / "gt.txt"
    # load gt: frame, id, x,y,w,h, conf, class, vis
    gt_by_frame = {}
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts)<9: continue
            frame = int(parts[0]); cls = int(parts[7]); vis = float(parts[8])
            if cls != 1: continue  # person only
            if vis < 0.2: continue
            x,y,bw,bh = map(float, parts[2:6])
            # clip to image bounds and skip tiny/outside
            x = max(0, min(x, w-1)); y = max(0, min(y, h-1))
            bw = max(1, min(bw, w - x)); bh = max(1, min(bh, h - y))
            if bw<10 or bh<20: continue
            gt_by_frame.setdefault(frame, []).append((x,y,bw,bh))
    converted = 0
    for img_file in sorted(img_src.glob("*.jpg")):
        frame_id = int(img_file.stem)
        boxes = gt_by_frame.get(frame_id, [])
        # destination
        dst_img = (IMG_VAL if is_val else IMG_TRAIN) / f"{seq_name}_{img_file.name}"
        dst_lbl = (LBL_VAL if is_val else LBL_TRAIN) / f"{seq_name}_{img_file.stem}.txt"
        # copy image (hardlink fallback to copy)
        try:
            shutil.copy2(img_file, dst_img)
        except Exception as e:
            print(f"copy fail {img_file}: {e}")
            continue
        # write label
        with open(dst_lbl,"w") as out:
            for x,y,bw,bh in boxes:
                xc = (x + bw/2) / w
                yc = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                # clamp
                xc = min(max(xc,0),1); yc=min(max(yc,0),1)
                out.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        # if no boxes, leave empty file (background image)
        converted+=1
    print(f"{seq_name}: {converted} frames -> {'val' if is_val else 'train'}")

def write_yaml():
    yaml_path = BASE / "configs" / "mot17.yaml"
    yaml_path.write_text(f"""# MOT17-YOLO for YOLO11
path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['person']
""")
    print(f"yaml: {yaml_path}")

def main():
    ensure_dirs()
    for seq in sorted(MOT_ROOT.iterdir()):
        if seq.name not in TRAIN_SEQS and seq.name not in VAL_SEQS:
            continue
        convert_seq(seq.name)
    # stats
    n_train = len(list(IMG_TRAIN.glob("*.jpg")))
    n_val = len(list(IMG_VAL.glob("*.jpg")))
    print(f"Done train={n_train} val={n_val} total={n_train+n_val}")
    write_yaml()

if __name__=="__main__":
    main()
