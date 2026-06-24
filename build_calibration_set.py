"""
Step 1: Build calibration set for LLM jury.
- Run one YOLO model on Arran validation images
- Crop each detection box from the image
- Match against ground truth to auto-label TP (real) / FP (fake)
- Select ~40 crops with ~1:3 real:fake ratio
"""

import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import csv, random

DATASET_DIR = Path(r"C:\Users\29775\Arran\yolo_dataset")
MODEL_DIR = Path(r"C:\Users\29775\Arran\yolo_runs")
MODELS = ["yolov8_seed0", "yolo11_seed0", "yolo26_seed0"]
OUT_DIR = Path(r"C:\Users\29775\calibration_set")
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {0: "roundhouse", 1: "shieling", 2: "smallcairn"}
CONF_THRESH = 0.01
IOU_MATCH = 0.3
IMG_SIZE = 512
PAD = 16


def load_gt(label_path, img_w, img_h):
    """Load YOLO-format ground truth, return list of (cls, x1, y1, x2, y2)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw/2) * img_w
            y1 = (cy - bh/2) * img_h
            x2 = (cx + bw/2) * img_w
            y2 = (cy + bh/2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def crop_box(img_arr, x1, y1, x2, y2, pad=PAD):
    h, w = img_arr.shape[:2]
    x1c = max(0, int(x1) - pad)
    y1c = max(0, int(y1) - pad)
    x2c = min(w, int(x2) + pad)
    y2c = min(h, int(y2) + pad)
    return img_arr[y1c:y2c, x1c:x2c]


if __name__ == "__main__":
    models = [YOLO(str(MODEL_DIR / m / "weights" / "best.pt")) for m in MODELS]
    print(f"Loaded {len(models)} models")

    val_dir = DATASET_DIR / "images" / "valid"
    lbl_dir = DATASET_DIR / "labels" / "valid"

    all_crops = []
    seen_boxes = []

    for img_path in sorted(val_dir.glob("*.png")):
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]

        gt_boxes = load_gt(lbl_dir / (img_path.stem + ".txt"), w, h)

        for model in models:
            results = model.predict(
                source=img, conf=CONF_THRESH, imgsz=IMG_SIZE,
                device=0, verbose=False, save=False,
            )

            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            for b in results[0].boxes:
                bx = b.xyxy[0].tolist()
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_name = CLASS_NAMES.get(cls_id, "unknown")

                # Dedup: skip if we already have a very similar box from another model
                dup = False
                for sb in seen_boxes:
                    if sb["src"] == img_path.name and sb["cls"] == cls_id:
                        if box_iou(bx, sb["xyxy"]) > 0.5:
                            dup = True
                            break
                if dup:
                    continue
                seen_boxes.append({"src": img_path.name, "cls": cls_id, "xyxy": bx})

                best_iou = 0
                for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
                    if gt_cls == cls_id:
                        iou = box_iou(bx, [gx1, gy1, gx2, gy2])
                        best_iou = max(best_iou, iou)

                label = "real" if best_iou >= IOU_MATCH else "fake"
                crop = crop_box(img, *bx)
                if crop.size == 0:
                    continue

                all_crops.append({
                    "crop": crop,
                    "cls": cls_name,
                    "conf": conf,
                    "label": label,
                    "src": img_path.name,
                    "iou": best_iou,
                })

    reals = [c for c in all_crops if c["label"] == "real"]
    fakes = [c for c in all_crops if c["label"] == "fake"]
    print(f"Total detections: {len(all_crops)} (real={len(reals)}, fake={len(fakes)})")

    # Target: ~40 crops, ratio ~1:3 (10 real, 30 fake)
    n_real = min(12, len(reals))
    n_fake = min(36, len(fakes))

    random.seed(42)
    selected_reals = random.sample(reals, n_real) if len(reals) > n_real else reals
    selected_fakes = random.sample(fakes, n_fake) if len(fakes) > n_fake else fakes
    selected = selected_reals + selected_fakes
    random.shuffle(selected)

    print(f"Selected: {len(selected)} (real={len(selected_reals)}, fake={len(selected_fakes)})")

    # Save crops and manifest
    manifest = []
    for i, item in enumerate(selected):
        fname = f"crop_{i:03d}_{item['cls']}_{item['label']}.png"
        Image.fromarray(item["crop"]).save(OUT_DIR / fname)
        manifest.append({
            "file": fname,
            "class": item["cls"],
            "label": item["label"],
            "conf": round(item["conf"], 3),
            "gt_iou": round(item["iou"], 3),
            "source_image": item["src"],
        })

    csv_path = OUT_DIR / "calibration_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)

    print(f"Saved {len(manifest)} crops to {OUT_DIR}")
    print(f"Manifest: {csv_path}")

    # Stats
    for cls in CLASS_NAMES.values():
        r = sum(1 for m in manifest if m["class"] == cls and m["label"] == "real")
        f = sum(1 for m in manifest if m["class"] == cls and m["label"] == "fake")
        print(f"  {cls}: real={r}, fake={f}")
