"""
Train YOLOv8, YOLO11, YOLO26 on Arran hillshade, 3 seeds each.
Then run consensus voting: a box counts only if >=2/3 seeds agree.
"""

import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_YAML = str(Path(r"C:\Users\29775\Arran\yolo_dataset\dataset.yaml"))
PROJECT_DIR  = Path(r"C:\Users\29775\Arran\yolo_runs")
SEEDS = [0, 1, 2]

MODELS = {
    "yolov8":  "yolov8n.pt",
    "yolo11":  "yolo11n.pt",
    "yolo26":  "yolo26n.pt",
}

TRAIN_ARGS = dict(
    data=DATASET_YAML,
    epochs=50,
    patience=15,
    imgsz=512,
    batch=16,
    device=0,
    workers=0,
    verbose=False,
    plots=True,
)

IOU_THRESH = 0.5
CONF_THRESH = 0.25
CONSENSUS_MIN = 2  # out of 3 seeds


# ── 1. Training ──────────────────────────────────────────────────────────────
def train_all():
    results_summary = {}
    for model_name, weights in MODELS.items():
        for seed in SEEDS:
            run_name = f"{model_name}_seed{seed}"
            run_dir = PROJECT_DIR / run_name
            best_pt = run_dir / "weights" / "best.pt"

            if best_pt.exists():
                print(f"[跳过] {run_name} (已训练: {best_pt})")
                results_summary[run_name] = str(best_pt)
                continue

            print(f"\n{'='*60}")
            print(f"训练: {run_name} ({weights}, seed={seed})")
            print(f"{'='*60}")

            model = YOLO(weights)
            metrics = model.train(
                **TRAIN_ARGS,
                seed=seed,
                name=run_name,
                project=str(PROJECT_DIR),
            )

            results_summary[run_name] = str(best_pt)
            map50 = metrics.box.map50 if hasattr(metrics, 'box') else 'N/A'
            print(f"[完成] {run_name}: mAP50 = {map50}")

    return results_summary


# ── 2. Predict with all seeds for one model version ─────────────────────────
def predict_all_seeds(model_name: str, img_dir: Path, conf: float = CONF_THRESH):
    """Return {image_stem: [list_of_detections_per_seed]}"""
    all_preds = defaultdict(list)

    for seed in SEEDS:
        run_name = f"{model_name}_seed{seed}"
        best_pt = PROJECT_DIR / run_name / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"  [!] 未找到权重: {best_pt}")
            continue

        model = YOLO(str(best_pt))
        results = model.predict(
            source=str(img_dir),
            conf=conf,
            iou=IOU_THRESH,
            imgsz=512,
            device=0,
            verbose=False,
            save=False,
        )

        for r in results:
            stem = Path(r.path).stem
            boxes = []
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    boxes.append({
                        "xyxy": b.xyxy[0].tolist(),
                        "conf": float(b.conf[0]),
                        "cls":  int(b.cls[0]),
                    })
            all_preds[stem].append(boxes)

    return all_preds


# ── 3. Consensus voting via box IoU matching ─────────────────────────────────
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def consensus_vote(all_preds: dict, min_agree: int = CONSENSUS_MIN,
                   iou_thresh: float = IOU_THRESH):
    """
    For each image, keep only boxes agreed by >= min_agree seeds.
    Strategy: use seed 0 boxes as anchors, count how many other seeds
    have a matching box (same class, IoU >= threshold).
    """
    consensus = {}

    for stem, seed_results in all_preds.items():
        if not seed_results or not seed_results[0]:
            consensus[stem] = []
            continue

        kept = []
        anchor_boxes = seed_results[0]

        for ab in anchor_boxes:
            votes = 1
            confs = [ab["conf"]]

            for other_seed_boxes in seed_results[1:]:
                matched = False
                for ob in other_seed_boxes:
                    if ob["cls"] != ab["cls"]:
                        continue
                    if box_iou(ab["xyxy"], ob["xyxy"]) >= iou_thresh:
                        votes += 1
                        confs.append(ob["conf"])
                        matched = True
                        break

            if votes >= min_agree:
                kept.append({
                    "xyxy": ab["xyxy"],
                    "cls": ab["cls"],
                    "conf_mean": float(np.mean(confs)),
                    "votes": votes,
                })

        # Also check boxes from other seeds not in anchor
        for seed_idx in range(1, len(seed_results)):
            for ob in seed_results[seed_idx]:
                already_covered = any(
                    ob["cls"] == k["cls"] and box_iou(ob["xyxy"], k["xyxy"]) >= iou_thresh
                    for k in kept
                )
                if already_covered:
                    continue

                votes = 1
                confs = [ob["conf"]]
                for other_idx in range(len(seed_results)):
                    if other_idx == seed_idx:
                        continue
                    for ob2 in seed_results[other_idx]:
                        if ob2["cls"] != ob["cls"]:
                            continue
                        if box_iou(ob["xyxy"], ob2["xyxy"]) >= iou_thresh:
                            votes += 1
                            confs.append(ob2["conf"])
                            break

                if votes >= min_agree:
                    kept.append({
                        "xyxy": ob["xyxy"],
                        "cls": ob["cls"],
                        "conf_mean": float(np.mean(confs)),
                        "votes": votes,
                    })

        consensus[stem] = kept

    return consensus


# ── 4. Evaluate consensus against ground truth ──────────────────────────────
def evaluate_consensus(consensus: dict, label_dir: Path, img_size: int = 500):
    """Compare consensus boxes against YOLO label files."""
    CLASS_NAMES = {0: "roundhouse", 1: "shieling", 2: "smallcairn"}
    tp, fp, fn = 0, 0, 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for stem, pred_boxes in consensus.items():
        lbl_file = label_dir / f"{stem}.txt"
        gt_boxes = []
        if lbl_file.exists():
            for line in lbl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - w/2) * img_size
                y1 = (cy - h/2) * img_size
                x2 = (cx + w/2) * img_size
                y2 = (cy + h/2) * img_size
                gt_boxes.append({"xyxy": [x1, y1, x2, y2], "cls": cls})

        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt or gb["cls"] != pb["cls"]:
                    continue
                iou = box_iou(pb["xyxy"], gb["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= IOU_THRESH and best_gt_idx >= 0:
                tp += 1
                per_class[pb["cls"]]["tp"] += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                per_class[pb["cls"]]["fp"] += 1

        for gi, gb in enumerate(gt_boxes):
            if gi not in matched_gt:
                fn += 1
                per_class[gb["cls"]]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"共识投票评估 (≥{CONSENSUS_MIN}/3 seeds)")
    print(f"{'='*50}")
    print(f"总体: TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    for cls_id in sorted(per_class.keys()):
        c = per_class[cls_id]
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0
        print(f"  {CLASS_NAMES.get(cls_id, cls_id)}: P={p:.3f} R={r:.3f} F1={f:.3f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time

    # Step 1: Train
    print("=" * 60)
    print("第一步：训练 3 版本 × 3 种子 = 9 个模型")
    print("=" * 60)

    start = time.time()
    train_all()
    train_time = time.time() - start
    print(f"\n训练总耗时: {train_time/60:.1f} 分钟")

    # Step 2: Consensus on validation set
    val_img_dir = Path(r"C:\Users\29775\Arran\yolo_dataset\images\valid")
    val_lbl_dir = Path(r"C:\Users\29775\Arran\yolo_dataset\labels\valid")

    print("\n" + "=" * 60)
    print("第二步：共识投票 (每个版本 3 种子)")
    print("=" * 60)

    all_results = {}
    for model_name in MODELS:
        print(f"\n--- {model_name} 共识 ---")
        preds = predict_all_seeds(model_name, val_img_dir)
        cons = consensus_vote(preds)
        metrics = evaluate_consensus(cons, val_lbl_dir)
        all_results[model_name] = metrics

        out_path = PROJECT_DIR / f"{model_name}_consensus.json"
        serializable = {k: v for k, v in cons.items()}
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  保存到: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"{'Model':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 60)
    for model_name, m in all_results.items():
        print(f"{model_name:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
