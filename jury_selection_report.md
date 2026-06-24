# Homework 10: LLM Jury Selection Report

## Calibration Set

- **48 crops** from Arran validation set YOLO detections
- **12 real** (TP matched ground truth, IoU >= 0.3) + **36 fake** (FP, no GT match)
- Ratio 1:3, matching high false-positive scenario
- Classes: roundhouse (3R/10F), shieling (5R/7F), smallcairn (4R/19F)

## N×2 Calibration Table

| Model | Prompt | Bal.Acc | 95% CI | TP | TN | FP | FN | Tokens | Latency |
|-------|--------|---------|--------|----|----|----|----|--------|---------|
| gpt-4o | bare | **0.708** | [0.558, 0.863] | 7 | 31 | 5 | 5 | 351 | 4.3s |
| qwen-vl-plus | criteria | 0.653 | [0.497, 0.782] | 7 | 27 | 9 | 5 | 310 | 2.2s |
| gpt-4o | criteria | 0.597 | [0.472, 0.742] | 5 | 30 | 6 | 7 | 477 | 4.4s |
| qwen-vl-plus | bare | 0.583 | [0.412, 0.749] | 6 | 25 | 11 | 6 | 172 | 1.7s |
| Doubao-1.5-vision-pro | bare | 0.542 | [0.500, 0.636] | 1 | 36 | 0 | 11 | 163 | 2.2s |
| Doubao-1.5-vision-pro | criteria | 0.542 | [0.500, 0.636] | 1 | 36 | 0 | 11 | 312 | 2.5s |

## Consistency Analysis

### Real detection (majority vote)
- **gpt-4o bare**: 7/12 real correctly identified (TPR = 0.583)
- **qwen-vl-plus criteria**: 7/12 real correctly identified (TPR = 0.583)
- **Doubao**: only 1/12 real detected (TPR = 0.083) — extremely conservative

### Fake detection (minority vote)
- **gpt-4o bare**: 31/36 fake correctly rejected (TNR = 0.861)
- **Doubao**: 36/36 fake rejected (TNR = 1.000) — but at the cost of missing almost all real sites
- **qwen-vl-plus bare**: 25/36 fake rejected (TNR = 0.694) — too permissive

### Prompt effect
- **gpt-4o**: bare (0.708) > criteria (0.597) — adding structured criteria hurt performance, possibly because the extra text distracted from visual judgment
- **qwen-vl-plus**: criteria (0.653) > bare (0.583) — benefited from explicit feature descriptions
- **Doubao**: no difference (0.542 both) — model struggles with this domain regardless of prompt

## Jury Selection

**I chose the top 3 by balanced accuracy:**

1. **gpt-4o + bare prompt** (BalAcc = 0.708)
2. **qwen-vl-plus + criteria prompt** (BalAcc = 0.653)
3. **gpt-4o + criteria prompt** (BalAcc = 0.597)

### Why these three?

1. **gpt-4o bare** is the clear winner — highest balanced accuracy with the tightest CI lower bound (0.558). It correctly identified 7/12 real sites while keeping FP low at 5/36. Its bare prompt works best because GPT-4o has strong built-in visual reasoning for this domain.

2. **qwen-vl-plus criteria** is the best non-GPT option — it matches gpt-4o's TPR (0.583) while being 2x faster and cheaper. The criteria prompt helps because qwen-vl-plus benefits from explicit archaeological feature descriptions to compensate for less domain knowledge.

3. **gpt-4o criteria** provides diversity in the jury — it uses the same strong base model but with a different prompt perspective. While its BalAcc is lower than bare, the combination of two GPT-4o prompts + one Qwen prompt gives the jury both "intuitive" and "criteria-guided" judgments.

### Why not Doubao?

Doubao-1.5-vision-pro is excluded because it is essentially a "reject everything" classifier (TP=1, FN=11). Its TNR=1.0 looks perfect, but a model that always says "fake" would achieve TNR=1.0 too. Its balanced accuracy of 0.542 is barely above random chance (0.5), making it unreliable as a jury member.

### Ensemble size: 3 vs 2?

- **3-member jury** (majority vote 2/3): more robust, tolerates one model error
- **2-member jury** (must agree): higher precision but lower recall
- Recommendation: use 3 members with 2/3 majority vote for the best balance
