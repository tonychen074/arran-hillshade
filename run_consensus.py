"""
Re-run consensus voting for all 3 model families on validation set.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT_DIR = Path(r"C:\Users\29775\Arran\yolo_runs")
VAL_IMG_DIR = Path(r"C:\Users\29775\Arran\yolo_dataset\images\valid")
VAL_LBL_DIR = Path(r"C:\Users\29775\Arran\yolo_dataset\labels\valid")

MODELS = ["yolov8", "yolo11", "yolo26"]
SEEDS = [0, 1, 2]
IOU_THRESH = 0.5
CONF_THRESH = 0.25
CONSENSUS_MIN = 2
CLASS_NAMES = {0: "roundhouse", 1: "shieling", 2: "smallcairn"}


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def predict_all_seeds(model_name):
    all_preds = defaultdict(list)
    for seed in SEEDS:
        best_pt = PROJECT_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
        model = YOLO(str(best_pt))
        results = model.predict(
            source=str(VAL_IMG_DIR), conf=CONF_THRESH, iou=IOU_THRESH,
            imgsz=512, device=0, verbose=False, save=False,
        )
        seed_box_count = 0
        for r in results:
            stem = Path(r.path).stem
            boxes = []
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    boxes.append({
                        "xyxy": b.xyxy[0].tolist(),
                        "conf": float(b.conf[0]),
                        "cls": int(b.cls[0]),
                    })
            seed_box_count += len(boxes)
            all_preds[stem].append(boxes)
        print(f"  seed{seed}: {seed_box_count} detections")
    return all_preds


def consensus_vote(all_preds):
    consensus = {}
    for stem, seed_results in all_preds.items():
        kept = []
        for si in range(len(seed_results)):
            for ob in seed_results[si]:
                covered = any(
                    ob["cls"] == k["cls"] and box_iou(ob["xyxy"], k["xyxy"]) >= IOU_THRESH
                    for k in kept
                )
                if covered:
                    continue
                votes = 1
                confs = [ob["conf"]]
                for oi in range(len(seed_results)):
                    if oi == si:
                        continue
                    for ob2 in seed_results[oi]:
                        if ob2["cls"] == ob["cls"] and box_iou(ob["xyxy"], ob2["xyxy"]) >= IOU_THRESH:
                            votes += 1
                            confs.append(ob2["conf"])
                            break
                if votes >= CONSENSUS_MIN:
                    kept.append({
                        "xyxy": ob["xyxy"],
                        "cls": ob["cls"],
                        "conf_mean": float(np.mean(confs)),
                        "votes": votes,
                    })
        consensus[stem] = kept
    return consensus


def evaluate(consensus):
    tp = fp = fn = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for stem, pred_boxes in consensus.items():
        lbl_file = VAL_LBL_DIR / f"{stem}.txt"
        gt_boxes = []
        if lbl_file.exists():
            for line in lbl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                x1, y1 = (cx - w/2) * 500, (cy - h/2) * 500
                x2, y2 = (cx + w/2) * 500, (cy + h/2) * 500
                gt_boxes.append({"xyxy": [x1, y1, x2, y2], "cls": cls})

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt or gb["cls"] != pb["cls"]:
                    continue
                iou = box_iou(pb["xyxy"], gb["xyxy"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= IOU_THRESH and best_gi >= 0:
                tp += 1
                per_class[pb["cls"]]["tp"] += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
                per_class[pb["cls"]]["fp"] += 1

        for gi, gb in enumerate(gt_boxes):
            if gi not in matched_gt:
                fn += 1
                per_class[gb["cls"]]["fn"] += 1

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "per_class": dict(per_class)}


if __name__ == "__main__":
    all_results = {}
    for model_name in MODELS:
        print(f"\n=== {model_name} ===")
        preds = predict_all_seeds(model_name)
        cons = consensus_vote(preds)
        total_cons = sum(len(v) for v in cons.values())
        print(f"  共识框: {total_cons}")
        metrics = evaluate(cons)
        all_results[model_name] = metrics

        with open(PROJECT_DIR / f"{model_name}_consensus.json", "w") as f:
            json.dump(cons, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Model':<10} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)
    for mn, m in all_results.items():
        print(f"{mn:<10} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
        for cid in sorted(m["per_class"]):
            c = m["per_class"][cid]
            cp = c["tp"]/(c["tp"]+c["fp"]) if (c["tp"]+c["fp"]) else 0
            cr = c["tp"]/(c["tp"]+c["fn"]) if (c["tp"]+c["fn"]) else 0
            cf = 2*cp*cr/(cp+cr) if (cp+cr) else 0
            print(f"  {CLASS_NAMES[cid]:<12} P={cp:.3f} R={cr:.3f} F1={cf:.3f}")
