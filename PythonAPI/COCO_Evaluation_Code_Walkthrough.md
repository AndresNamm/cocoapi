# COCO Evaluation Code Walkthrough

This document maps the conceptual steps of COCO evaluation to the specific lines of code in `pycocotools/cocoeval.py`.

## 1. Per Image Processing & IoU Calculation

**Concept:** For each image, calculate IoU between every detection and every ground truth. Detections are sorted by confidence and limited by `maxDets` before this step.

**Code Reference:** `computeIoU` method in `cocoeval.py`.

*   **Sorting by confidence:**
    ```python
    # Line 213
    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    ```

*   **Limiting by `maxDets`:**
    ```python
    # Lines 215-216
    if len(dt) > p.maxDets[-1]:
        dt=dt[0:p.maxDets[-1]]
    ```

*   **Calculating IoU:**
    ```python
    # Line 229
    ious = maskUtils.iou(d,g,iscrowd)
    ```

## 2. Matching Strategy (TP/FP Assignment)

**Concept:** Match detections to GT objects based on IoU and confidence. A detection is a TP if it matches a GT with IoU >= threshold and is the highest confidence match for that GT.

**Code Reference:** `evaluateImg` method in `cocoeval.py`.

*   **Sorting detections (again, per image):**
    ```python
    # Line 295
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    ```

*   **Matching Loop:**
    Iterates through IoU thresholds (`tind`) and detections (`dind`).
    ```python
    # Lines 308-309
    for tind, t in enumerate(p.iouThrs):
        for dind, d in enumerate(dt):
    ```

*   **Finding Best Match:**
    Checks if GT is already matched (`gtm`) or if IoU is below threshold.
    ```python
    # Lines 315-323
    # if this gt already matched, and not a crowd, continue
    if gtm[tind,gind]>0 and not iscrowd[gind]:
        continue
    # ...
    # continue to next gt unless better match made
    if ious[dind,gind] < iou:
        continue
    # if match successful and best so far, store appropriately
    iou=ious[dind,gind]
    m=gind
    ```

*   **Assigning Match:**
    If a valid match `m` is found, it's recorded.
    ```python
    # Lines 327-328
    dtIg[tind,dind] = gtIg[m]
    dtm[tind,dind]  = gt[m]['id']
    gtm[tind,m]     = d['id']
    ```

## 3. Recall Calculation

**Concept:** Concatenate all detections, sort globally by confidence, calculate cumulative TP, and compute Recall.

**Code Reference:** `accumulate` method in `cocoeval.py`.

*   **Concatenate Detections:**
    ```python
    # Line 380
    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
    ```

*   **Global Sort:**
    ```python
    # Lines 385-386
    inds = np.argsort(-dtScores, kind='mergesort')
    dtScoresSorted = dtScores[inds]
    ```

*   **Cumulative TP & FP:**
    ```python
    # Lines 394-395
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
    ```

*   **Recall Formula:**
    ```python
    # Line 398
    rc = tp / npig
    ```

*   **Final Recall Value:**
    Takes the last value of the recall array.
    ```python
    # Line 403
    recall[t,k,a,m] = rc[-1]
    ```

## 4. Average Precision (AP) Calculation

**Concept:** Calculate Precision at each point, interpolate to 101 fixed recall thresholds, and average.

**Code Reference:** `accumulate` and `summarize` methods in `cocoeval.py`.

*   **Precision Formula:**
    ```python
    # Line 399
    pr = tp / (fp+tp+np.spacing(1))
    ```

*   **Smoothing (Interpolation Step 1):**
    Ensures the precision curve is monotonically decreasing (looking from right to left).
    ```python
    # Lines 408-410
    for i in range(nd-1, 0, -1):
        if pr[i] > pr[i-1]:
            pr[i-1] = pr[i]
    ```

*   **101-Point Interpolation (Step 2):**
    Finds indices for 101 recall thresholds (`p.recThrs`) and samples precision.
    ```python
    # Lines 412-415
    inds = np.searchsorted(rc, p.recThrs, side='left')
    try:
        for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
            # ...
    ```

*   **Averaging (in `summarize`):**
    Calculates the mean of the precision values across categories/IoUs/areas.
    ```python
    # Line 466 (inside _summarize inner function)
    mean_s = np.mean(s[s>-1])
    ```
