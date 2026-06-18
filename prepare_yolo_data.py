"""
Prepare YOLO dataset from Arran hillshade tiles + labels.
Converts GeoTIFF hillshade to 3-channel PNG, organizes into YOLO directory structure.
"""

import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio

BASE = Path(r"C:\Users\29775\Arran")
HILLSHADE_DIR = BASE / "Hillshade"
LABEL_DIR = BASE / "DTM" / "labels"
DATASET_DIR = BASE / "yolo_dataset"

def tif_to_png(tif_path: Path, png_path: Path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
    img = Image.fromarray(arr, "L").convert("RGB")
    img.save(png_path)

def prepare():
    for split in ["train", "valid"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        src_lbl_dir = LABEL_DIR / ("valid" if split == "valid" else split)
        label_files = sorted(src_lbl_dir.glob("*.txt"))

        converted = 0
        skipped = 0
        for lbl_file in label_files:
            stem = lbl_file.stem
            hillshade_tif = HILLSHADE_DIR / f"{stem}_hillshade16.tif"

            if not hillshade_tif.exists():
                skipped += 1
                continue

            png_out = img_dir / f"{stem}.png"
            if not png_out.exists():
                tif_to_png(hillshade_tif, png_out)

            lbl_out = lbl_dir / lbl_file.name
            if not lbl_out.exists():
                shutil.copy2(lbl_file, lbl_out)

            converted += 1

        print(f"{split}: {converted} 图像+标签, 跳过 {skipped} (无对应山影)")

    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(f"""path: {DATASET_DIR}
train: images/train
val: images/valid

names:
  0: roundhouse
  1: shieling
  2: smallcairn
""", encoding="utf-8")
    print(f"\ndataset.yaml -> {yaml_path}")

if __name__ == "__main__":
    prepare()
