"""
Per-class mAP@50 and mAP@50-95 evaluation for all 9 models.
Averages across 3 seeds per model family.
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(r"C:\Users\29775\Arran\yolo_runs")
DATASET_YAML = str(Path(r"C:\Users\29775\Arran\yolo_dataset\dataset.yaml"))

MODELS = ["yolov8", "yolo11", "yolo26"]
SEEDS = [0, 1, 2]
CLASS_NAMES = ["roundhouse", "shieling", "smallcairn"]


def eval_single(model_name, seed):
    best_pt = PROJECT_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
    if not best_pt.exists():
        return None
    model = YOLO(str(best_pt))
    metrics = model.val(data=DATASET_YAML, imgsz=512, device=0, verbose=False, plots=False)

    result = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50": [float(x) for x in metrics.box.ap50],
        "per_class_map50_95": [float(x) for x in metrics.box.ap],
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    return result


if __name__ == "__main__":
    all_results = {}

    for model_name in MODELS:
        seed_results = []
        for seed in SEEDS:
            print(f"评估 {model_name}_seed{seed}...", end=" ", flush=True)
            r = eval_single(model_name, seed)
            if r:
                seed_results.append(r)
                print(f"mAP50={r['map50']:.3f}  mAP50-95={r['map50_95']:.3f}")
            else:
                print("跳过")
        all_results[model_name] = seed_results

    # ── Per-seed detail table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("各种子详细结果")
    print(f"{'='*80}")
    print(f"{'Model':<18} {'mAP50':>7} {'mAP50-95':>9}  ", end="")
    for cn in CLASS_NAMES:
        print(f" {cn[:10]+' @50':>15}", end="")
    print()
    print("-" * 80)

    for model_name in MODELS:
        for si, r in enumerate(all_results[model_name]):
            label = f"{model_name}_seed{si}"
            print(f"{label:<18} {r['map50']:>7.3f} {r['map50_95']:>9.3f}  ", end="")
            for ci in range(len(CLASS_NAMES)):
                v = r["per_class_map50"][ci] if ci < len(r["per_class_map50"]) else 0
                print(f" {v:>15.3f}", end="")
            print()
        print()

    # ── Averaged across seeds ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("三版本对比 (3种子均值 ± 标准差)")
    print(f"{'='*80}")

    header = f"{'Model':<10} {'mAP@50':>12} {'mAP@50-95':>14}"
    for cn in CLASS_NAMES:
        header += f"  {cn[:10]+'@50':>14}"
    print(header)
    print("-" * 80)

    summary = {}
    for model_name in MODELS:
        seeds = all_results[model_name]
        if not seeds:
            continue

        map50s = [r["map50"] for r in seeds]
        map50_95s = [r["map50_95"] for r in seeds]

        row = f"{model_name:<10} {np.mean(map50s):>5.3f}±{np.std(map50s):.3f} {np.mean(map50_95s):>6.3f}±{np.std(map50_95s):.3f}"

        per_class_summary = {}
        for ci, cn in enumerate(CLASS_NAMES):
            vals = [r["per_class_map50"][ci] for r in seeds if ci < len(r["per_class_map50"])]
            row += f"  {np.mean(vals):>5.3f}±{np.std(vals):.3f}"
            per_class_summary[cn] = {"mean": np.mean(vals), "std": np.std(vals)}

        print(row)
        summary[model_name] = {
            "map50": {"mean": np.mean(map50s), "std": np.std(map50s)},
            "map50_95": {"mean": np.mean(map50_95s), "std": np.std(map50_95s)},
            "per_class": per_class_summary,
        }

    # ── Per-class mAP@50-95 ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("逐类 mAP@50-95 (3种子均值)")
    print(f"{'='*80}")
    print(f"{'Model':<10}", end="")
    for cn in CLASS_NAMES:
        print(f"  {cn:>14}", end="")
    print()
    print("-" * 56)

    for model_name in MODELS:
        seeds = all_results[model_name]
        print(f"{model_name:<10}", end="")
        for ci, cn in enumerate(CLASS_NAMES):
            vals = [r["per_class_map50_95"][ci] for r in seeds if ci < len(r["per_class_map50_95"])]
            print(f"  {np.mean(vals):>14.3f}", end="")
        print()

    # ── Winner summary ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    best_model = max(summary, key=lambda m: summary[m]["map50"]["mean"])
    print(f"总体 mAP@50 最高: {best_model} ({summary[best_model]['map50']['mean']:.3f})")

    for cn in CLASS_NAMES:
        best = max(summary, key=lambda m: summary[m]["per_class"][cn]["mean"])
        val = summary[best]["per_class"][cn]["mean"]
        print(f"  {cn} 最佳: {best} (mAP@50 = {val:.3f})")
