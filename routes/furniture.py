"""
routes/furniture.py  — FIXED
------------------------------
Bug 3 fixed: MultiDetector() was created INSIDE the async function on every
             request. Loading YOLOv8-Large takes ~15-30s per call — requests
             timed out before any detection was returned.
             Now uses the shared singleton from app.state (set by main.py).

Bug 4 fixed: detect(mode="furniture") already filters via FURNITURE_CLASSES.
             The second manual filter here used a different label set and
             silently discarded valid detections.
"""

from fastapi import APIRouter, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse
import cv2

from utils.image import require_image, b64

router = APIRouter(prefix="/furniture", tags=["Furniture Detection"])

LABEL_MAP = {
    "couch":        "Sofa / Couch",
    "dining table": "Dining Table",
    "potted plant": "Potted Plant",
    "cell phone":   "Mobile Phone",
    "wine glass":   "Wine Glass",
}

BOX_COLORS = {
    "chair":        (255, 180,  60),
    "couch":        (255, 120,  40),
    "bed":          (100, 200, 255),
    "dining table": ( 80, 220, 120),
    "tv":           (200,  80, 255),
    "refrigerator": ( 60, 220, 210),
    "default":      (130, 170, 255),
}


def _color(label: str):
    return BOX_COLORS.get(label, BOX_COLORS["default"])


def _friendly(label: str) -> str:
    return LABEL_MAP.get(label, label.title())


@router.post("/", summary="Detect furniture in an interior photo")
async def detect_furniture(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.30, ge=0.10, le=0.95),
):
    # FIX: pull singleton set in main.py startup — no new model load
    detector = request.app.state.detector

    try:
        image = require_image(await file.read())
    except Exception as e:
        return JSONResponse({"error": f"Cannot read image: {e}"}, status_code=400)

    try:
        # FIX: no second filter — detect() already handles FURNITURE_CLASSES
        detections = detector.detect(image, confidence, mode="furniture")
    except Exception as e:
        return JSONResponse({"error": f"Detection failed: {e}"}, status_code=500)

    for d in detections:
        d["display_name"] = _friendly(d.get("label", "object"))
        x1, y1, x2, y2   = d["bbox"]
        d["area_px"]      = (x2 - x1) * (y2 - y1)

    by_type: dict[str, int] = {}
    for d in detections:
        key = d["display_name"]
        by_type[key] = by_type.get(key, 0) + 1

    annotated = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = _color(d.get("label", "default"))
        label_text = f"{d['display_name']}  {d['confidence']:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(annotated, label_text, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return JSONResponse({
        "total_items":     len(detections),
        "unique_types":    len(by_type),
        "by_type":         by_type,
        "detections":      detections,
        "annotated_image": b64(annotated),
        "original_image":  b64(image),
    })