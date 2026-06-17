"""
Single vs 16-direction hillshade comparison.
Shows structures that disappear under a single illumination direction.
"""

import sys
sys.path.insert(0, r"C:\Users\29775")

import numpy as np
import rasterio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv

from hillshade16 import compute_slope_aspect, hillshade_single, normalize_percentile, multidirectional_hillshade

TILE = "192035-638177.tif"
DTM_PATH  = Path(r"C:\Users\29775\Arran\DTM\images") / TILE
CSV_PATH  = Path(r"C:\Users\29775\Arran\DTM\train_annotations.csv")
OUT_PATH  = Path(r"C:\Users\29775\hillshade_comparison.png")

# ── 1. load DEM ──────────────────────────────────────────────────────────────
with rasterio.open(DTM_PATH) as src:
    dem = src.read(1).astype(np.float32)
    nodata = src.nodata
    cellsize = abs(src.transform[0])

valid = dem != nodata if nodata is not None else np.ones(dem.shape, bool)
valid &= dem > -9000
dem[~valid] = dem[valid].mean()

slope, aspect = compute_slope_aspect(dem, cellsize)

# ── 2. render three versions ──────────────────────────────────────────────────
def render(arr, valid_mask):
    n = normalize_percentile(arr, valid_mask)
    out = (n * 255).astype(np.uint8)
    out[~valid_mask] = 0
    return out

# Standard NW (315°) – most common default
hs_nw  = render(hillshade_single(slope, aspect, azimuth_deg=315), valid)
# Perpendicular to NW: NE (45°) – structures parallel to NW illumination vanish here
hs_ne  = render(hillshade_single(slope, aspect, azimuth_deg=45),  valid)
# 16-direction average
hs_16  = render(multidirectional_hillshade(dem, cellsize, n=16),   valid)

# ── 3. load annotation boxes for this tile ───────────────────────────────────
boxes = []
with open(CSV_PATH, encoding="utf-8") as f:
    for row in csv.reader(f):
        if not row or TILE not in row[0]:
            continue
        # format: path,x1,y1,x2,y2,class
        try:
            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            label = row[5] if len(row) > 5 else ""
            boxes.append((x1, y1, x2, y2, label))
        except (ValueError, IndexError):
            pass

# ── 4. build comparison image ─────────────────────────────────────────────────
COLOURS = {
    "round house": "#FF4444",
    "small cairn":  "#44AAFF",
    "shieling hut": "#44FF88",
}
DEFAULT_COL = "#FFFF00"

PANEL_W, PANEL_H = dem.shape[1], dem.shape[0]
GAP = 12
LABEL_H = 36
TOTAL_W = PANEL_W * 3 + GAP * 2
TOTAL_H = PANEL_H + LABEL_H

canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), (30, 30, 30))

panels = [
    (hs_nw,  "Single direction  315° NW  (standard)"),
    (hs_ne,  "Single direction   45° NE  (perpendicular)"),
    (hs_16,  "16-direction average  (this script)"),
]

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
except OSError:
    font = ImageFont.load_default()
    font_sm = font

for col_idx, (arr, title) in enumerate(panels):
    x_off = col_idx * (PANEL_W + GAP)

    img = Image.fromarray(arr, "L").convert("RGB")
    draw = ImageDraw.Draw(img)

    for (x1, y1, x2, y2, label) in boxes:
        col = COLOURS.get(label.lower().strip(), DEFAULT_COL)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)

    canvas.paste(img, (x_off, LABEL_H))

    # panel title
    draw_c = ImageDraw.Draw(canvas)
    draw_c.text((x_off + 6, 6), title, fill="#FFFFFF", font=font)

# legend
draw_c = ImageDraw.Draw(canvas)
lx = TOTAL_W - 280
ly = LABEL_H + 8
for label, col in COLOURS.items():
    draw_c.rectangle([lx, ly, lx+14, ly+14], outline=col, fill=col)
    draw_c.text((lx + 18, ly), label, fill="#FFFFFF", font=font_sm)
    ly += 20

canvas.save(OUT_PATH, "PNG")
print(f"保存到: {OUT_PATH}")
print(f"标注框数量: {len(boxes)}")
print(f"图像尺寸: {TOTAL_W} x {TOTAL_H} px")
