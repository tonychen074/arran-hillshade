"""
Convert Arran CSV annotations to YOLO format.
Input:  images/xxx.tif, x1, y1, x2, y2, class
Output: labels/xxx.txt with lines: class_id cx cy w h (normalized 0-1)
"""

import csv
from pathlib import Path
from collections import defaultdict

IMG_W, IMG_H = 500, 500

CLASS_MAP = {
    "roundhouse": 0,
    "shieling": 1,
    "smallcairn": 2,
}

BASE = Path(r"C:\Users\29775\Arran\DTM")

def convert_csv(csv_path: Path, label_dir: Path):
    label_dir.mkdir(parents=True, exist_ok=True)

    boxes = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            img_name = Path(row[0]).stem
            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            cls = row[5].strip().lower()
            if cls not in CLASS_MAP:
                print(f"  跳过未知类别: {cls}")
                continue

            cx = ((x1 + x2) / 2.0) / IMG_W
            cy = ((y1 + y2) / 2.0) / IMG_H
            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H

            boxes[img_name].append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    for stem, lines in boxes.items():
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(boxes), sum(len(v) for v in boxes.values())


# Convert train and valid
for split, csv_name in [("train", "train_annotations.csv"), ("valid", "valid_annotations.csv")]:
    csv_path = BASE / csv_name
    label_dir = BASE / "labels" / split
    n_imgs, n_boxes = convert_csv(csv_path, label_dir)
    print(f"{split}: {n_boxes} 个框 -> {n_imgs} 个 txt 文件 ({label_dir})")

# Write classes.txt
classes_path = BASE / "labels" / "classes.txt"
classes_path.write_text("roundhouse\nshieling\nsmallcairn\n", encoding="utf-8")
print(f"\n类别文件: {classes_path}")
print("类别映射: roundhouse=0, shieling=1, smallcairn=2")
