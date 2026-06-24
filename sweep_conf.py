"""
Sweep confidence thresholds: show how low conf → high recall,
then consensus voting recovers precision.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT_DIR = Path(r"C:\Users\29775\Arran\yolo_runs")
VAL_IMG_DIR = Path(r"C:\Users\29775\Arran\yolo_dataset\images\valid")
VAL_LBL_DIR = Path(r"C:\Users\29775\Arran\yolo_dataset\labels\valid")

MODELS = ["yolov8", "yolo11", "yolo26"]
SEEDS = [0, 1, 2]
CONF_LEVELS = [0.25, 0.15, 0.10, 0.05]
IOU_THRESH = 0.5
CONSENSUS_MIN = 2
CLASS_NAMES = {0: "roundhouse", 1: "shieling", 2: "smallcairn"}


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def load_gt():
    gt = {}
    for lbl_file in VAL_LBL_DIR.glob("*.txt"):
        stem = lbl_file.stem
        boxes = []
        for line in lbl_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            boxes.append({
                "xyxy": [(cx-w/2)*500, (cy-h/2)*500, (cx+w/2)*500, (cy+h/2)*500],
                "cls": cls,
            })
        gt[stem] = boxes
    return gt


def evaluate(pred_boxes_per_img, gt):
    tp = fp = fn = 0
    for stem, gt_boxes in gt.items():
        preds = pred_boxes_per_img.get(stem, [])
        matched = set()
        for pb in preds:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched or gb["cls"] != pb["cls"]:
                    continue
                iou = box_iou(pb["xyxy"], gb["xyxy"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= IOU_THRESH and best_gi >= 0:
                tp += 1
                matched.add(best_gi)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn,
            "n_pred": tp+fp}


def predict_model(model_path, conf):
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(VAL_IMG_DIR), conf=conf, iou=IOU_THRESH,
        imgsz=512, device=0, verbose=False, save=False,
    )
    preds = {}
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
        preds[stem] = boxes
    return preds


def consensus_vote(all_seed_preds):
    consensus = {}
    stems = set()
    for sp in all_seed_preds:
        stems.update(sp.keys())
    for stem in stems:
        seed_results = [sp.get(stem, []) for sp in all_seed_preds]
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
                for oi in range(len(seed_results)):
                    if oi == si:
                        continue
                    for ob2 in seed_results[oi]:
                        if ob2["cls"] == ob["cls"] and box_iou(ob["xyxy"], ob2["xyxy"]) >= IOU_THRESH:
                            votes += 1
                            break
                if votes >= CONSENSUS_MIN:
                    kept.append(ob)
        consensus[stem] = kept
    return consensus


if __name__ == "__main__":
    gt = load_gt()
    total_gt = sum(len(v) for v in gt.values())
    print(f"验证集: {len(gt)} 图, {total_gt} 真值框\n")

    # ── 1. Single best seed per model at each conf ───────────────────────
    print("=" * 90)
    print("第一部分：单模型单种子（取各版本最佳种子）随置信度变化")
    print("=" * 90)
    print(f"{'Model':<10} {'Conf':>6} {'#Pred':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 90)

    for model_name in MODELS:
        for conf in CONF_LEVELS:
            best_f1 = -1
            best_metrics = None
            for seed in SEEDS:
                pt = PROJECT_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
                preds = predict_model(pt, conf)
                m = evaluate(preds, gt)
                if m["F1"] > best_f1:
                    best_f1 = m["F1"]
                    best_metrics = m
            m = best_metrics
            print(f"{model_name:<10} {conf:>6.2f} {m['n_pred']:>7} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['P']:>7.3f} {m['R']:>7.3f} {m['F1']:>7.3f}")
        print()

    # ── 2. Consensus at each conf ────────────────────────────────────────
    print("=" * 90)
    print("第二部分：共识投票 (≥2/3 seeds) 随置信度变化")
    print("=" * 90)
    print(f"{'Model':<10} {'Conf':>6} {'#Pred':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 90)

    for model_name in MODELS:
        for conf in CONF_LEVELS:
            seed_preds = []
            for seed in SEEDS:
                pt = PROJECT_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
                preds = predict_model(pt, conf)
                seed_preds.append(preds)
            cons = consensus_vote(seed_preds)
            m = evaluate(cons, gt)
            print(f"{model_name:<10} {conf:>6.2f} {m['n_pred']:>7} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['P']:>7.3f} {m['R']:>7.3f} {m['F1']:>7.3f}")
        print()

    # ── 3. Cross-model mega consensus ────────────────────────────────────
    print("=" * 90)
    print("第三部分：全模型大陪审团 (9个模型, ≥6/9 同意)")
    print("=" * 90)
    print(f"{'Conf':>6} {'MinVote':>8} {'#Pred':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 90)

    for conf in CONF_LEVELS:
        all_preds = []
        for model_name in MODELS:
            for seed in SEEDS:
                pt = PROJECT_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
                preds = predict_model(pt, conf)
                all_preds.append(preds)

        for min_vote in [3, 5, 6, 7]:
            cons = {}
            stems = set()
            for sp in all_preds:
                stems.update(sp.keys())
            for stem in stems:
                seed_results = [sp.get(stem, []) for sp in all_preds]
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
                        for oi in range(len(seed_results)):
                            if oi == si:
                                continue
                            for ob2 in seed_results[oi]:
                                if ob2["cls"] == ob["cls"] and box_iou(ob["xyxy"], ob2["xyxy"]) >= IOU_THRESH:
                                    votes += 1
                                    break
                        if votes >= min_vote:
                            kept.append(ob)
                cons[stem] = kept
            m = evaluate(cons, gt)
            print(f"{conf:>6.2f} {min_vote:>5}/9  {m['n_pred']:>7} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['P']:>7.3f} {m['R']:>7.3f} {m['F1']:>7.3f}")
        print()
