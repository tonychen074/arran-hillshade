"""
Homework 10: LLM Jury Harness
- Step 2: Criteria JSON for archaeological site types
- Step 3: Jury harness (3 models × 2 prompts × N crops)
- Step 4: Balanced accuracy, bootstrap 95% CI, avg tokens, avg latency
- Step 5: Jury selection analysis
"""

import base64, csv, json, time, os, random
import numpy as np
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────
API_KEY = "sk-a7EWElPCkW3qIjWDRwNtKvBsin84SuWO379wfyqxoxsH0aVy"
BASE_URL = "https://www.dmxapi.cn/v1"

CALIBRATION_DIR = Path(r"C:\Users\29775\calibration_set")
MANIFEST_CSV = CALIBRATION_DIR / "calibration_manifest.csv"
CACHE_FILE = Path(r"C:\Users\29775\jury_cache.json")
RESULTS_CSV = Path(r"C:\Users\29775\jury_results.csv")

MODELS = [
    "gpt-4o",
    "qwen-vl-plus",
    "Doubao-1.5-vision-pro-32k",
]

# ── Step 2: Criteria JSON ─────────────────────────────────────────────
CRITERIA_JSON = {
    "roundhouse": {
        "positive": [
            "Circular or near-circular enclosure outline visible as a ring",
            "Diameter typically 5-15 meters",
            "Defined wall footprint or foundation ring, sometimes with entrance gap",
            "Located on relatively flat or gently sloping ground",
        ],
        "negative_confusers": [
            "Natural circular depressions or peat hags",
            "Tree-throw pits or root plates",
            "Circular noise artifacts from hillshade rendering",
            "Modern circular features (water tanks, silos)",
        ],
    },
    "shieling": {
        "positive": [
            "Small rectangular or sub-rectangular structure outline",
            "Typically 3-6 meters long, 2-3 meters wide",
            "Often found in upland/moorland areas",
            "May appear as paired parallel walls or a simple enclosure",
            "Sometimes clustered in groups",
        ],
        "negative_confusers": [
            "Natural rock outcrops with linear edges",
            "Drainage ditches or field boundaries",
            "Modern sheep pens or fank remnants",
            "Random shadow patterns on steep slopes",
        ],
    },
    "smallcairn": {
        "positive": [
            "Small mound or raised circular feature",
            "Diameter typically 2-8 meters",
            "Appears as a bright spot (higher elevation) surrounded by darker ground",
            "Relatively symmetrical dome shape in hillshade",
        ],
        "negative_confusers": [
            "Natural boulders or glacial erratics",
            "Modern field clearance stone piles",
            "Hillock or natural terrain bump",
            "Hillshade artifacts at ridge crests",
        ],
    },
}

CRITERIA_STR = json.dumps(CRITERIA_JSON, indent=2)

# ── Prompts ───────────────────────────────────────────────────────────
PROMPTS = {
    "bare": (
        "You are an archaeological site detector reviewing LiDAR hillshade imagery. "
        "This cropped image shows a detection from an automated detector. "
        "The detector claims this is a {det_class}. "
        "Judge whether this detection is a REAL archaeological feature or a FALSE POSITIVE. "
        "Respond with JSON only: {{\"verdict\": \"real\" or \"fake\", \"confidence\": 0-100, \"reason\": \"brief explanation\"}}"
    ),
    "criteria": (
        "You are an archaeological site detector reviewing LiDAR hillshade imagery. "
        "This cropped image shows a detection from an automated detector. "
        "The detector claims this is a {det_class}. "
        "\n\nUse the following expert criteria to judge:\n"
        "{criteria}\n\n"
        "Judge whether this detection is a REAL archaeological feature or a FALSE POSITIVE. "
        "Respond with JSON only: {{\"verdict\": \"real\" or \"fake\", \"confidence\": 0-100, \"reason\": \"brief explanation\"}}"
    ),
}


def load_manifest():
    rows = []
    with open(MANIFEST_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def judge(client, model, prompt_text, image_path, cache):
    prompt_hash = hash(prompt_text) & 0xFFFFFFFF
    cache_key = f"{model}|{prompt_hash}|{image_path.name}"
    if cache_key in cache:
        return cache[cache_key]

    b64 = image_to_base64(image_path)
    ext = image_path.suffix.lower().strip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        latency = time.time() - t0
        text = resp.choices[0].message.content
        tokens_in = resp.usage.prompt_tokens if resp.usage else 0
        tokens_out = resp.usage.completion_tokens if resp.usage else 0

        # Strip markdown code fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean)
            verdict = parsed.get("verdict", "").lower().strip()
        except json.JSONDecodeError:
            if "real" in text.lower():
                verdict = "real"
            elif "fake" in text.lower() or "false" in text.lower():
                verdict = "fake"
            else:
                verdict = "unknown"
            parsed = {"verdict": verdict, "raw": text}

    except Exception as e:
        latency = time.time() - t0
        verdict = "error"
        parsed = {"verdict": "error", "error": str(e)}
        tokens_in, tokens_out = 0, 0

    result = {
        "verdict": verdict,
        "response": parsed,
        "latency": round(latency, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }

    cache[cache_key] = result
    save_cache(cache)
    return result


def balanced_accuracy(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    pos = sum(1 for t in y_true if t == 1)
    neg = sum(1 for t in y_true if t == 0)
    tpr = tp / pos if pos > 0 else 0
    tnr = tn / neg if neg > 0 else 0
    return (tpr + tnr) / 2


def bootstrap_ci(y_true, y_pred, n_boot=2000, ci=0.95):
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        if len(set(yt)) < 2:
            continue
        scores.append(balanced_accuracy(yt, yp))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return lo, hi


if __name__ == "__main__":
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    manifest = load_manifest()
    cache = load_cache()

    print(f"Calibration set: {len(manifest)} crops")
    print(f"Models: {MODELS}")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Cache entries: {len(cache)}")
    print()

    # ── Three-level loop ──────────────────────────────────────────────
    all_results = []

    for model in MODELS:
        for prompt_name, prompt_template in PROMPTS.items():
            print(f"\n{'='*60}")
            print(f"Model: {model} | Prompt: {prompt_name}")
            print(f"{'='*60}")

            y_true = []
            y_pred = []
            latencies = []
            tokens_list = []

            for i, crop_info in enumerate(manifest):
                img_path = CALIBRATION_DIR / crop_info["file"]
                det_class = crop_info["class"]
                gt_label = crop_info["label"]  # real or fake

                if prompt_name == "criteria":
                    cls_criteria = CRITERIA_JSON.get(det_class, {})
                    prompt_text = prompt_template.format(
                        det_class=det_class,
                        criteria=json.dumps(cls_criteria, indent=2),
                    )
                else:
                    prompt_text = prompt_template.format(det_class=det_class)

                result = judge(client, model, prompt_text, img_path, cache)

                y_true.append(1 if gt_label == "real" else 0)
                pred_v = result["verdict"]
                y_pred.append(1 if pred_v == "real" else 0)
                latencies.append(result["latency"])
                tokens_list.append(result["tokens_in"] + result["tokens_out"])

                status = "OK" if (pred_v == gt_label) else "MISS"
                print(f"  [{i+1:2d}/{len(manifest)}] {crop_info['file']}: "
                      f"GT={gt_label}, Pred={pred_v} [{status}] "
                      f"({result['latency']:.1f}s)")

            ba = balanced_accuracy(y_true, y_pred)
            ci_lo, ci_hi = bootstrap_ci(y_true, y_pred)
            avg_tokens = np.mean(tokens_list) if tokens_list else 0
            avg_latency = np.mean(latencies) if latencies else 0

            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

            print(f"\n  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
            print(f"  Balanced Accuracy: {ba:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
            print(f"  Avg Tokens: {avg_tokens:.0f}  Avg Latency: {avg_latency:.1f}s")

            all_results.append({
                "model": model,
                "prompt": prompt_name,
                "balanced_acc": round(ba, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "avg_tokens": round(avg_tokens, 1),
                "avg_latency": round(avg_latency, 2),
            })

    # ── Save results table ────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f"\nResults saved to {RESULTS_CSV}")

    # ── Step 5: Summary table ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Model':<30} {'Prompt':<10} {'BalAcc':>8} {'95% CI':>16} {'Tokens':>8} {'Latency':>8}")
    print(f"{'='*80}")
    for r in sorted(all_results, key=lambda x: -x["balanced_acc"]):
        ci_str = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"{r['model']:<30} {r['prompt']:<10} {r['balanced_acc']:>8.3f} {ci_str:>16} "
              f"{r['avg_tokens']:>8.0f} {r['avg_latency']:>7.1f}s")
