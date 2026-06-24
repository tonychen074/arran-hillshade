"""
Run trained YOLO ensemble on Kintyre/Galloway hillshade tiles.
1. Slice large hillshade GeoTIFFs into 512x512 patches
2. Run 9 models (3 versions x 3 seeds) on each patch
3. Mega-jury consensus voting (>=5/9)
4. Convert pixel coords -> BNG coords via GeoTIFF transform
5. Cross-reference with Canmore known sites for dedup
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import json, csv, time

# ── Config ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Users\29775\Arran\yolo_runs")
CANMORE_CSV = Path(r"C:\Users\29775\Canmore_Points\canmore_dedup_reference.csv")

REGIONS = {
    "Kintyre": Path(r"C:\Users\29775\Scotland_DTM\Kintyre_Hillshade"),
    "Galloway": Path(r"C:\Users\29775\Scotland_DTM\Galloway_Hillshade"),
}

MODELS = ["yolov8", "yolo11", "yolo26"]
SEEDS = [0, 1, 2]
PATCH_SIZE = 512
STRIDE = 384  # overlap = 128px
CONF = 0.15
IOU_THRESH = 0.5
MIN_VOTES = 5  # out of 9
DEDUP_DIST_M = 50.0  # match radius in meters

CLASS_NAMES = {0: "roundhouse", 1: "shieling", 2: "smallcairn"}
CLASS_MAP_CANMORE = {"roundhouse": "roundhouse", "shieling": "shieling", "cairn": "smallcairn"}


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def load_canmore():
    df = pd.read_csv(CANMORE_CSV)
    sites = []
    for _, row in df.iterrows():
        sites.append({
            "x": float(row["XCOORD"]),
            "y": float(row["YCOORD"]),
            "cls": row["det_class"],
            "name": row["NMRSNAME"],
            "canmore_id": int(row["CANMOREID"]),
        })
    return sites


def slice_and_predict(hs_path, models_loaded):
    """Slice a hillshade GeoTIFF into patches and run all 9 models."""
    with rasterio.open(hs_path) as src:
        h, w = src.height, src.width
        transform = src.transform

        if src.count == 1:
            band = src.read(1)
            img_full = np.stack([band, band, band], axis=-1)  # grayscale -> 3ch
        else:
            img_full = np.transpose(src.read()[:3], (1, 2, 0))

    all_detections = []  # list of {xyxy_pixel, cls, conf, model_idx}

    patches = []
    for y0 in range(0, h, STRIDE):
        for x0 in range(0, w, STRIDE):
            ye = min(y0 + PATCH_SIZE, h)
            xe = min(x0 + PATCH_SIZE, w)
            if ye - y0 < 64 or xe - x0 < 64:
                continue
            patch = img_full[y0:ye, x0:xe]
            # Pad if smaller than PATCH_SIZE
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            patches.append((x0, y0, patch))

    print(f"    {len(patches)} patches", end="", flush=True)

    for mi, model in enumerate(models_loaded):
        batch_imgs = [p[2] for p in patches]
        results = model.predict(
            source=batch_imgs, conf=CONF, iou=IOU_THRESH,
            imgsz=PATCH_SIZE, device=0, verbose=False, save=False,
            batch=16,
        )
        for pi, r in enumerate(results):
            x0, y0, _ = patches[pi]
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    bx = b.xyxy[0].tolist()
                    # Offset to full-image coordinates
                    bx_full = [bx[0]+x0, bx[1]+y0, bx[2]+x0, bx[3]+y0]
                    all_detections.append({
                        "xyxy": bx_full,
                        "cls": int(b.cls[0]),
                        "conf": float(b.conf[0]),
                        "model_idx": mi,
                    })
        print(".", end="", flush=True)
    print(f" {len(all_detections)} raw detections")

    return all_detections, transform


def mega_jury(detections, min_votes=MIN_VOTES):
    """Cross-model consensus: keep boxes with >= min_votes models agreeing."""
    kept = []
    for det in detections:
        covered = any(
            det["cls"] == k["cls"] and box_iou(det["xyxy"], k["xyxy"]) >= IOU_THRESH
            for k in kept
        )
        if covered:
            continue
        voters = {det["model_idx"]}
        confs = [det["conf"]]
        for det2 in detections:
            if det2["model_idx"] in voters:
                continue
            if det2["cls"] == det["cls"] and box_iou(det["xyxy"], det2["xyxy"]) >= IOU_THRESH:
                voters.add(det2["model_idx"])
                confs.append(det2["conf"])
        if len(voters) >= min_votes:
            kept.append({
                "xyxy": det["xyxy"],
                "cls": det["cls"],
                "conf_mean": float(np.mean(confs)),
                "votes": len(voters),
            })
    return kept


def pixel_to_bng(xyxy, transform):
    """Convert pixel bbox to BNG coordinates (center point)."""
    cx_px = (xyxy[0] + xyxy[2]) / 2
    cy_px = (xyxy[1] + xyxy[3]) / 2
    bng_x = transform[2] + cx_px * transform[0] + cy_px * transform[1]
    bng_y = transform[5] + cx_px * transform[3] + cy_px * transform[4]
    return bng_x, bng_y


def dedup_with_canmore(detections, canmore_sites, dist_thresh=DEDUP_DIST_M):
    """Match detections against Canmore known sites."""
    matched = []
    novel = []
    for det in detections:
        cls_name = CLASS_NAMES.get(det["cls"], "unknown")
        best_dist = float("inf")
        best_site = None
        for site in canmore_sites:
            canmore_cls = CLASS_MAP_CANMORE.get(site["cls"])
            if canmore_cls != cls_name:
                continue
            dx = det["bng_x"] - site["x"]
            dy = det["bng_y"] - site["y"]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < best_dist:
                best_dist = dist
                best_site = site

        det_info = {
            "bng_x": det["bng_x"],
            "bng_y": det["bng_y"],
            "class": cls_name,
            "conf": det["conf_mean"],
            "votes": det["votes"],
        }

        if best_dist <= dist_thresh and best_site:
            det_info["status"] = "known"
            det_info["canmore_id"] = best_site["canmore_id"]
            det_info["canmore_name"] = best_site["name"]
            det_info["dist_m"] = round(best_dist, 1)
            matched.append(det_info)
        else:
            det_info["status"] = "novel"
            det_info["nearest_dist_m"] = round(best_dist, 1) if best_dist < float("inf") else -1
            novel.append(det_info)

    return matched, novel


if __name__ == "__main__":
    t0 = time.time()

    # Load all 9 models
    print("Loading 9 models...")
    models_loaded = []
    for model_name in MODELS:
        for seed in SEEDS:
            pt = PROJECT_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
            models_loaded.append(YOLO(str(pt)))
    print(f"  Loaded {len(models_loaded)} models")

    # Load Canmore reference
    canmore = load_canmore()
    print(f"  Canmore reference: {len(canmore)} sites")

    all_results = []

    for region_name, hs_dir in REGIONS.items():
        print(f"\n{'='*60}")
        print(f"Region: {region_name}")
        print(f"{'='*60}")

        hs_files = sorted(hs_dir.glob("*_hillshade16.tif"))
        print(f"  {len(hs_files)} hillshade tiles")

        region_matched = []
        region_novel = []

        for fi, hs_path in enumerate(hs_files):
            print(f"\n  [{fi+1}/{len(hs_files)}] {hs_path.name}")

            detections, transform = slice_and_predict(hs_path, models_loaded)

            if not detections:
                print("    No detections")
                continue

            consensus = mega_jury(detections)
            print(f"    Consensus: {len(consensus)} boxes (>={MIN_VOTES}/9 votes)")

            # Convert to BNG
            for det in consensus:
                det["bng_x"], det["bng_y"] = pixel_to_bng(det["xyxy"], transform)

            # Dedup
            matched, novel = dedup_with_canmore(consensus, canmore)
            region_matched.extend(matched)
            region_novel.extend(novel)

            for cls_id, cls_name in CLASS_NAMES.items():
                n_cls = sum(1 for d in consensus if d["cls"] == cls_id)
                if n_cls:
                    print(f"      {cls_name}: {n_cls}")

        print(f"\n  --- {region_name} Summary ---")
        print(f"  Known (matched Canmore): {len(region_matched)}")
        print(f"  Novel (potential new):   {len(region_novel)}")

        # Save results
        out_dir = Path(r"C:\Users\29775\inference_results")
        out_dir.mkdir(exist_ok=True)

        for label, data in [("known", region_matched), ("novel", region_novel)]:
            if data:
                out_csv = out_dir / f"{region_name.lower()}_{label}.csv"
                keys = data[0].keys()
                with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(data)
                print(f"  Saved: {out_csv}")

        all_results.append({
            "region": region_name,
            "known": len(region_matched),
            "novel": len(region_novel),
        })

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY (elapsed: {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['region']}: {r['known']} known + {r['novel']} novel = {r['known']+r['novel']} total")
    total_known = sum(r["known"] for r in all_results)
    total_novel = sum(r["novel"] for r in all_results)
    print(f"  TOTAL: {total_known} known + {total_novel} novel = {total_known+total_novel}")
