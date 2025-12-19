# How Average Recall (AR) is Calculated in COCO Evaluation

## Overview

Average Recall (AR) in COCO evaluation represents the maximum recall achievable given a fixed number of detections per image, averaged across IoU thresholds and object categories. This document explains the step-by-step calculation process after IoU values are computed.

---

## Step-by-Step Calculation Process

### 1. **IoU Computation** (`evaluate()` method)

After IoU values are calculated for all detection-ground truth pairs, the evaluation performs matching:

```python
# For each IoU threshold (default: 0.50, 0.55, ..., 0.95)
for tind, t in enumerate(p.iouThrs):
    for dind, d in enumerate(dt):  # For each detection
        for gind, g in enumerate(gt):  # For each ground truth
            # Match detection to ground truth if IoU >= threshold
            if ious[dind, gind] >= iou:
                # Store the match
```

**Output per image/category:**
- `dtMatches`: [T×D] matrix - which ground truth each detection matches at each IoU threshold
- `gtMatches`: [T×G] matrix - which detection each ground truth matches at each IoU threshold
- `dtScores`: [D] array - confidence scores of detections
- `gtIgnore`: [G] array - whether each ground truth should be ignored

---

### 2. **Accumulation Across Images** (`accumulate()` method)

#### 2.1 Concatenate Results
For each (category, area range, maxDets) combination:

```python
# Gather all detection scores across all images
dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

# Sort detections by confidence score (highest first)
inds = np.argsort(-dtScores, kind='mergesort')

# Concatenate matches and ignore flags across all images
dtm = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet] for e in E], axis=1)[:,inds]
gtIg = np.concatenate([e['gtIgnore'] for e in E])
```

#### 2.2 Count Ground Truth Objects
```python
npig = np.count_nonzero(gtIg == 0)  # Number of non-ignored ground truths
```

#### 2.3 Identify True Positives and False Positives
For each IoU threshold:

```python
# True Positive: detection matched to GT and not ignored
tps = np.logical_and(dtm, np.logical_not(dtIg))

# False Positive: detection not matched to GT and not ignored
fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
```

#### 2.4 Calculate Cumulative TP and FP
```python
tp_sum = np.cumsum(tps, axis=1)  # Cumulative true positives
fp_sum = np.cumsum(fps, axis=1)  # Cumulative false positives
```

After sorting by confidence, as we go through detections:
- TP count increases when a detection matches a ground truth
- FP count increases when a detection doesn't match any ground truth

---

### 3. **Recall Calculation** (Key Step for AR)

For each IoU threshold `t`:

```python
nd = len(tp)  # Number of detections
rc = tp / npig  # Recall at each detection point
```

**Recall formula:**
$$\text{Recall} = \frac{\text{True Positives}}{\text{Total Ground Truth Objects}}$$

The **maximum recall** at this IoU threshold is stored:

```python
if nd:
    recall[t, k, a, m] = rc[-1]  # Last value = maximum recall
else:
    recall[t, k, a, m] = 0
```

**Key insight:** `rc[-1]` is the recall when ALL detections (up to maxDets) are considered. This represents the maximum possible recall given the detection set.

---

### 4. **Array Dimensions**

The recall array has dimensions **[T×K×A×M]**:
- **T**: Number of IoU thresholds (default: 10, from 0.50 to 0.95)
- **K**: Number of categories
- **A**: Number of area ranges (4: all, small, medium, large)
- **M**: Number of maxDets settings (3: 1, 10, 100)

---

### 5. **Summarization** (`summarize()` method)

#### 5.1 Extract Relevant Recall Values

For a specific AR metric (e.g., AR @maxDets=100, area=all):

```python
# dimension of recall: [TxKxAxM]
s = self.eval['recall']

# Filter by area range (e.g., 'all', 'small', 'medium', 'large')
s = s[:, :, aind, mind]

# s now contains recall values for:
# - All IoU thresholds (0.50 to 0.95)
# - All categories
# - Specific area range
# - Specific maxDets value
```

#### 5.2 Calculate Average Recall

```python
mean_s = np.mean(s[s > -1])
```

**This averages across:**
1. **All IoU thresholds** (0.50, 0.55, ..., 0.95) - 10 values
2. **All categories** (person, car, etc.)
3. **All images** (already aggregated in step 2)

---

## Example Calculation

Let's say you have:
- **100 images**
- **5 categories**
- **10 IoU thresholds** (0.50 to 0.95)
- **maxDets = 100**
- **area = 'all'**

### Step-by-step:

1. **For each of the 10 IoU thresholds:**
   - Match detections to ground truths
   - Calculate maximum recall across all 100 images and 5 categories
   - Example: at IoU=0.50, recall might be 0.65

2. **Recall array might look like:**
   ```
   IoU=0.50: recall = 0.65
   IoU=0.55: recall = 0.60
   IoU=0.60: recall = 0.55
   ...
   IoU=0.95: recall = 0.20
   ```

3. **Average across all thresholds:**
   ```
   AR = (0.65 + 0.60 + 0.55 + ... + 0.20) / 10 = 0.417
   ```

This gives you **AR @[IoU=0.50:0.95 | area=all | maxDets=100] = 0.417**

---

## Key Formulas

### Maximum Recall at IoU threshold t:
$$\text{Recall}_t = \frac{\sum_{i=1}^{N} \text{TP}_i(t)}{\sum_{i=1}^{N} \text{GT}_i}$$

Where:
- $\text{TP}_i(t)$ = true positives for image $i$ at IoU threshold $t$
- $\text{GT}_i$ = ground truth objects in image $i$
- $N$ = number of images

### Average Recall:
$$\text{AR} = \frac{1}{T} \sum_{t=1}^{T} \text{Recall}_t$$

Where:
- $T$ = number of IoU thresholds (typically 10)

---

## Important Notes

1. **Maximum Recall**: AR uses the **maximum** recall achievable when considering up to `maxDets` detections per image (sorted by confidence). This is why `rc[-1]` is used.

2. **No Precision Penalty**: Unlike AP, AR doesn't penalize false positives directly. It only measures what percentage of ground truth objects can be recalled.

3. **IoU Averaging**: AR averages across strict IoU thresholds (0.50 to 0.95), making it more challenging than metrics that only use IoU=0.50.

4. **Area Ranges**:
   - Small: area < 32²
   - Medium: 32² < area < 96²
   - Large: area > 96²

5. **maxDets Values**:
   - maxDets=1: Only the highest-confidence detection per image
   - maxDets=10: Up to 10 detections per image
   - maxDets=100: Up to 100 detections per image

---

## Interpretation Example

From your evaluation output:
```
Average Recall (AR) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = 0.417
```

**This means:**
- When allowing up to 100 detections per image
- Averaged across IoU thresholds from 0.50 to 0.95
- For objects of all sizes
- Your model can recall **41.7%** of all ground truth objects

**Why AR differs from AP:**
- AR (0.417) > AP (0.320) in your case
- AR measures maximum possible recall (best case)
- AP measures precision-recall tradeoff (penalizes false positives)
- High AR with lower AP means the model finds many objects but also produces false positives

---

## Code Reference

The key calculation happens in `cocoeval.py`:

```python
# Line ~391: Calculate recall for each IoU threshold
rc = tp / npig  # tp = cumulative true positives, npig = total ground truths
recall[t,k,a,m] = rc[-1]  # Maximum recall (when all detections considered)

# Line ~456: Average across IoU thresholds
mean_s = np.mean(s[s>-1])  # Average all non-negative recall values
```
