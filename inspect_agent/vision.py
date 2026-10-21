"""OpenCV 5 perception tools used by the agent."""
from __future__ import annotations
import numpy as np
import cv2


def quality(img: np.ndarray) -> dict:
    """Image-quality measurements that drive the agent's next action."""
    lap = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_64F)
    return {
        "brightness": float(img.mean()),
        "sharpness": float(lap.var()),
        "saturated": float((img >= 250).mean()),
        "dynamic_range": float(np.percentile(img, 99) - np.percentile(img, 1)),
    }


def detect(img: np.ndarray, sens: float = 1.0, zoom: float = 1.0) -> dict:
    """Find dark surface anomalies: background-normalise, threshold, morphology, contours."""
    g = img.astype(np.float32)
    k = min(99, int(21 * zoom) | 1)
    bg = cv2.medianBlur(img, k).astype(np.float32)
    diff = np.clip(bg - g, 0, None)  # defects are darker than local background
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    sigma = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) + 1e-3
    thr = np.median(diff) + (4.0 / sens) * sigma
    mask = (diff > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 * zoom:
            continue
        H, W = img.shape
        m = 6
        if x <= m or y <= m or x + w >= W - m or y + h >= H - m:
            if area < 40 * zoom:   # small blobs touching the frame edge = median-filter border artefacts
                continue
        comp = (labels == i)
        strength = float(diff[comp].mean() / sigma)
        pts = np.column_stack(np.nonzero(comp)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        elong = max(rw, rh) / max(1.0, min(rw, rh))
        kind = "scratch" if (elong > 3 and max(rw, rh) > 20) else "pit"
        cands.append({"bbox": (int(x), int(y), int(w), int(h)), "area": int(area), "strength": round(strength, 2), "kind": kind})
    cands.sort(key=lambda c: -c["strength"] * np.sqrt(c["area"]))
    best = cands[0] if cands else None
    # confidence that a real defect exists: strength margin over the noise floor
    score = 0.0 if not best else float(1 - np.exp(-max(0.0, best["strength"] - 4.0) * min(best["area"], 200) / 60))
    return {"candidates": cands[:5], "best": best, "defect_score": round(score, 3), "noise_sigma": round(sigma, 3), "mask": mask}


def enhance(img: np.ndarray) -> np.ndarray:
    """Software fallback when re-capture isn't possible: CLAHE local contrast."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def annotate(img: np.ndarray, det: dict, verdict: str) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = {"PASS": (60, 180, 75), "REJECT": (40, 40, 220), "ESCALATE": (0, 170, 255)}.get(verdict, (200, 200, 200))
    for c in det["candidates"][:3]:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 2)
        cv2.putText(vis, f"{c['kind']} {c['strength']:.1f}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(vis, verdict, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis
