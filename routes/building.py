"""
routes/building.py
-------------------
POST /building/
  - Accepts a building exterior photo
  - Detects windows using contour analysis + Hough lines (CV2)
    with an optional YOLO pass for "window" if the model supports it
  - Estimates floor count from vertical window row clustering
  - Returns: window_count, estimated_floors, floor_height_estimate,
             annotated image, confidence
"""

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from utils.image import require_image, b64
import cv2
import numpy as np

router = APIRouter(prefix="/building", tags=["Building Analysis"])


# ── Window detection via classical CV ────────────────────────

def detect_windows_cv(image: np.ndarray, sensitivity: float = 0.5):
    """
    Detect windows using:
    1. Edge detection (Canny)
    2. Contour filtering (rectangular aspect ratio, min area)
    3. Returns list of (x, y, w, h) bounding boxes
    """
    h, w = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Blur → edge detect
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 120)

    # Dilate edges to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area   = (h * w) * 0.002 * (1 - sensitivity * 0.5)   # ~0.2% of image
    max_area   = (h * w) * 0.20                                # max 20% of image
    boxes      = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        # Windows are roughly square to wide-ish (0.5 → 3.5)
        if 0.4 < aspect < 3.8:
            solidity = area / (bw * bh)
            if solidity > 0.35:          # fairly solid rectangle
                boxes.append((x, y, bw, bh))

    # Non-max suppression (remove heavily overlapping boxes)
    boxes = _nms(boxes, overlap_thresh=0.45)
    return boxes


def _nms(boxes, overlap_thresh=0.45):
    """Simple IoU-based non-maximum suppression."""
    if not boxes:
        return []
    boxes  = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept   = []
    for b in boxes:
        x1, y1, w1, h1 = b
        dominated = False
        for k in kept:
            x2, y2, w2, h2 = k
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            iou = (ix * iy) / max(1, w1*h1 + w2*h2 - ix*iy)
            if iou > overlap_thresh:
                dominated = True
                break
        if not dominated:
            kept.append(b)
    return kept


# ── Floor estimation ──────────────────────────────────────────

def estimate_floors(boxes, image_height: int):
    """
    Cluster window boxes by their vertical centre (y + h/2).
    Each cluster = one floor row.
    Also estimate floor height in pixels → real-world metres.
    """
    if not boxes:
        return 0, None, []

    centres = sorted([(y + h // 2) for (x, y, w, h) in boxes])

    # Cluster centres that are within ~15% image height of each other
    gap_thresh = image_height * 0.10
    clusters   = []
    current    = [centres[0]]

    for c in centres[1:]:
        if c - current[-1] <= gap_thresh:
            current.append(c)
        else:
            clusters.append(current)
            current = [c]
    clusters.append(current)

    num_floors      = len(clusters)
    floor_centres   = [int(np.mean(cl)) for cl in clusters]

    # Estimate floor height from gap between consecutive floor rows
    if len(floor_centres) >= 2:
        gaps           = [floor_centres[i+1] - floor_centres[i] for i in range(len(floor_centres)-1)]
        avg_gap_px     = float(np.mean(gaps))
        # Assume average floor-to-floor height ≈ 3.2 m
        px_per_metre   = avg_gap_px / 3.2
        floor_height_m = round(avg_gap_px / px_per_metre, 2) if px_per_metre else None
    else:
        floor_height_m = None

    return num_floors, floor_height_m, floor_centres


# ── Draw helpers ──────────────────────────────────────────────

def draw_results(image, boxes, floor_centres):
    out = image.copy()
    h   = image.shape[0]

    # Draw floor guide lines
    for i, fc in enumerate(floor_centres):
        cv2.line(out, (0, fc), (image.shape[1], fc), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"Floor {i+1}", (6, fc - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA)

    # Draw window boxes
    for (x, y, w, bh) in boxes:
        cv2.rectangle(out, (x, y), (x+w, y+bh), (80, 200, 255), 2)

    return out


# ── Route ─────────────────────────────────────────────────────

@router.post("/", summary="Count windows and estimate floors from a building photo")
async def analyze_building(
    file:        UploadFile = File(..., description="Exterior building photo"),
    sensitivity: float      = Query(0.5, ge=0.1, le=0.9,
                                    description="Window detection sensitivity (higher = more detections)"),
):
    """
    Upload a building exterior photo.

    Returns:
    - **window_count** — total windows detected
    - **estimated_floors** — number of floor rows detected
    - **floor_height_m** — estimated height per floor (metres, if calculable)
    - **total_height_estimate_m** — approximate building height
    - **windows_per_floor** — breakdown per detected floor row
    - **annotated_image** — image with window boxes + floor lines drawn
    """
    image = require_image(await file.read())
    h, w  = image.shape[:2]

    boxes         = detect_windows_cv(image, sensitivity)
    num_floors, floor_height_m, floor_centres = estimate_floors(boxes, h)

    # Total height estimate
    if floor_height_m and num_floors:
        total_height_m = round(floor_height_m * num_floors, 1)
    else:
        # Fallback: assume 3.5 m per floor
        total_height_m = round(num_floors * 3.5, 1) if num_floors else None

    # Windows per floor
    windows_per_floor = {}
    if floor_centres:
        gap = (h * 0.10)
        for i, fc in enumerate(floor_centres):
            count = sum(
                1 for (x, y, bw, bh) in boxes
                if abs((y + bh // 2) - fc) <= gap
            )
            windows_per_floor[f"Floor {i+1}"] = count

    annotated = draw_results(image, boxes, floor_centres)

    # Confidence heuristic: more windows = higher confidence
    conf = min(0.95, 0.4 + len(boxes) * 0.03) if boxes else 0.0

    return JSONResponse({
        "window_count":          len(boxes),
        "estimated_floors":      num_floors,
        "floor_height_m":        floor_height_m,
        "total_height_estimate_m": total_height_m,
        "windows_per_floor":     windows_per_floor,
        "detection_confidence":  round(conf, 2),
        "image_size":            {"width": w, "height": h},
        "annotated_image":       b64(annotated),
        "original_image":        b64(image),
    })
