"""
routes/waste.py
----------------
POST /waste/
  - Detects waste items in a photo
  - Classifies each item into material type:
      Plastic | Paper/Cardboard | Glass | Metal | Organic | Electronic | Hazardous | Mixed
  - Returns recyclability verdict, disposal tips, and recyclability rate
"""

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from utils.image import require_image, b64
import cv2

router = APIRouter(prefix="/waste", tags=["Waste Identification"])


# ── Material classification rules ────────────────────────────
# Maps COCO/YOLO labels → material type + recyclability info

WASTE_RULES: dict[str, dict] = {
    # ── Plastic ──────────────────────────────────────────────
    "bottle":         {"material": "Plastic",         "recyclable": True,  "score": 0.85},
    "plastic bag":    {"material": "Plastic",         "recyclable": False, "score": 0.10},
    "cup":            {"material": "Plastic/Paper",   "recyclable": True,  "score": 0.60},
    "straw":          {"material": "Plastic",         "recyclable": False, "score": 0.05},
    "fork":           {"material": "Plastic/Metal",   "recyclable": True,  "score": 0.55},
    "knife":          {"material": "Metal",           "recyclable": True,  "score": 0.75},
    "spoon":          {"material": "Plastic/Metal",   "recyclable": True,  "score": 0.55},

    # ── Paper / Cardboard ─────────────────────────────────────
    "book":           {"material": "Paper",           "recyclable": True,  "score": 0.90},
    "newspaper":      {"material": "Paper",           "recyclable": True,  "score": 0.95},
    "cardboard":      {"material": "Cardboard",       "recyclable": True,  "score": 0.95},
    "pizza box":      {"material": "Cardboard",       "recyclable": False, "score": 0.20},  # grease contaminated

    # ── Glass ────────────────────────────────────────────────
    "wine glass":     {"material": "Glass",           "recyclable": True,  "score": 0.88},
    "vase":           {"material": "Glass/Ceramic",   "recyclable": False, "score": 0.30},

    # ── Electronics ──────────────────────────────────────────
    "cell phone":     {"material": "Electronic",      "recyclable": True,  "score": 0.70},
    "laptop":         {"material": "Electronic",      "recyclable": True,  "score": 0.70},
    "keyboard":       {"material": "Electronic",      "recyclable": True,  "score": 0.65},
    "remote":         {"material": "Electronic",      "recyclable": True,  "score": 0.60},
    "tv":             {"material": "Electronic",      "recyclable": True,  "score": 0.65},

    # ── Organic ───────────────────────────────────────────────
    "banana":         {"material": "Organic",         "recyclable": True,  "score": 1.00},  # compost
    "apple":          {"material": "Organic",         "recyclable": True,  "score": 1.00},
    "orange":         {"material": "Organic",         "recyclable": True,  "score": 1.00},
    "broccoli":       {"material": "Organic",         "recyclable": True,  "score": 1.00},
    "carrot":         {"material": "Organic",         "recyclable": True,  "score": 1.00},
    "sandwich":       {"material": "Organic/Paper",   "recyclable": False, "score": 0.30},
    "pizza":          {"material": "Organic/Mixed",   "recyclable": False, "score": 0.20},
    "hot dog":        {"material": "Organic",         "recyclable": False, "score": 0.30},
    "cake":           {"material": "Organic",         "recyclable": False, "score": 0.30},
    "donut":          {"material": "Organic",         "recyclable": False, "score": 0.30},

    # ── Metal / Cans ──────────────────────────────────────────
    "scissors":       {"material": "Metal",           "recyclable": True,  "score": 0.80},
    "can":            {"material": "Metal",           "recyclable": True,  "score": 0.90},

    # ── Clothing / Textile ────────────────────────────────────
    "handbag":        {"material": "Textile/Leather", "recyclable": True,  "score": 0.50},
    "backpack":       {"material": "Textile",         "recyclable": True,  "score": 0.50},
    "tie":            {"material": "Textile",         "recyclable": True,  "score": 0.50},
    "suitcase":       {"material": "Plastic/Textile", "recyclable": False, "score": 0.35},

    # ── Hazardous ────────────────────────────────────────────
    "battery":        {"material": "Hazardous",       "recyclable": True,  "score": 0.75},  # special drop-off
}

# Disposal tips per material
DISPOSAL_TIPS: dict[str, str] = {
    "Plastic":         "Rinse and place in plastic recycling bin. Remove caps if different plastic type.",
    "Plastic/Paper":   "Separate plastic lid from paper cup. Check local recycling rules.",
    "Paper":           "Dry paper goes in paper recycling. Shred sensitive documents first.",
    "Cardboard":       "Flatten boxes before placing in recycling bin.",
    "Glass":           "Rinse bottles. Place in glass recycling bin. Do NOT mix with ceramics.",
    "Metal":           "Rinse cans, crush if space is limited. Place in metal/can recycling.",
    "Organic":         "Compost in green bin or home composter. Great for garden fertiliser.",
    "Organic/Mixed":   "Non-compostable mixed food waste — general waste bin.",
    "Organic/Paper":   "If heavily soiled, general waste. Clean paper portions can be recycled.",
    "Electronic":      "Take to an e-waste drop-off centre. Do NOT put in general waste.",
    "Hazardous":       "Take to a hazardous waste collection point. Never put in regular bins.",
    "Textile":         "Donate if good condition, else textile recycling bank.",
    "Textile/Leather": "Donate or take to textile recycling. Avoid landfill.",
    "Glass/Ceramic":   "Ceramics are NOT recyclable in glass bins — general waste.",
    "Plastic/Metal":   "Separate components if possible. Metal parts in metal bin.",
    "Plastic/Textile": "Hard to recycle — check specialist recyclers.",
    "Mixed":           "General waste if cannot be separated.",
    "Metal/Glass":     "Separate and recycle each component individually.",
}

# Box colours per material (BGR)
MATERIAL_COLORS: dict[str, tuple] = {
    "Plastic":         (255, 165,  0),
    "Plastic/Paper":   (200, 180,  0),
    "Paper":           ( 60, 180, 255),
    "Cardboard":       (100, 140, 200),
    "Glass":           ( 80, 220, 180),
    "Metal":           (160, 160, 200),
    "Organic":         ( 60, 200,  80),
    "Organic/Mixed":   (100, 180,  80),
    "Organic/Paper":   (100, 200, 120),
    "Electronic":      (200,  80, 240),
    "Hazardous":       ( 20,  20, 240),
    "Textile":         (220, 120, 180),
    "Textile/Leather": (200, 100, 140),
    "Glass/Ceramic":   ( 60, 200, 160),
    "Plastic/Metal":   (180, 140,  60),
    "Plastic/Textile": (200, 140,  80),
    "Mixed":           (140, 140, 140),
}


def classify(label: str) -> dict:
    """Return waste classification for a detected label."""
    key = label.lower()
    if key in WASTE_RULES:
        return WASTE_RULES[key]
    # Fallback heuristics
    if any(k in key for k in ["box", "wrap", "bag", "pack"]):
        return {"material": "Cardboard", "recyclable": True, "score": 0.80}
    if any(k in key for k in ["can", "tin", "metal"]):
        return {"material": "Metal", "recyclable": True, "score": 0.90}
    if any(k in key for k in ["glass", "jar", "bottle"]):
        return {"material": "Glass", "recyclable": True, "score": 0.85}
    return {"material": "Mixed", "recyclable": False, "score": 0.30}


@router.post("/", summary="Identify waste type and recyclability from a photo")
async def identify_waste(
    file:       UploadFile = File(..., description="Photo of waste / rubbish"),
    confidence: float      = Query(0.25, ge=0.10, le=0.95,
                                   description="Detection confidence threshold"),
):
    """
    Upload a photo of waste.

    Returns:
    - **by_material** — breakdown of detected waste by material type
    - **recyclable_count / non_recyclable_count**
    - **recyclability_rate** — percentage of items that are recyclable
    - **disposal_tips** — how to dispose of each material found
    - **special_notes** — warnings for hazardous items
    - **detections** — full per-item detection list
    - **annotated_image** — image with coloured boxes per material
    """
    from models.detector import MultiDetector
    detector = MultiDetector(model_size="l")

    image      = require_image(await file.read())
    raw        = detector.detect(image, confidence, mode="waste")

    # ── Enrich detections with waste classification ───────────
    detections = []
    for d in raw:
        info = classify(d.get("label", ""))
        d["material"]           = info["material"]
        d["recyclable"]         = info["recyclable"]
        d["recyclability_score"]= info["score"]
        d["disposal_tip"]       = DISPOSAL_TIPS.get(info["material"], "Check local waste guidelines.")
        if info["material"] == "Hazardous":
            d["special_note"] = "⚠️ HAZARDOUS — Take to specialist collection point only."
        detections.append(d)

    # ── Summary stats ─────────────────────────────────────────
    total           = len(detections)
    rec_count       = sum(1 for d in detections if d["recyclable"])
    non_rec_count   = total - rec_count
    rec_rate        = round(rec_count / total * 100, 1) if total else 0.0
    avg_score       = round(sum(d["recyclability_score"] for d in detections) / total, 2) if total else 0.0

    # ── By material ───────────────────────────────────────────
    by_material: dict[str, dict] = {}
    for d in detections:
        m = d["material"]
        if m not in by_material:
            by_material[m] = {
                "count":      0,
                "recyclable": d["recyclable"],
                "disposal_tip": d["disposal_tip"],
                "items":      [],
            }
        by_material[m]["count"] += 1
        by_material[m]["items"].append(d.get("label", "unknown"))

    # ── Special notes (hazardous only) ───────────────────────
    special_notes = [
        d["special_note"] for d in detections if "special_note" in d
    ]

    # ── Draw annotated image ──────────────────────────────────
    annotated = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color           = MATERIAL_COLORS.get(d["material"], (140, 140, 140))
        rec_icon        = "✓" if d["recyclable"] else "✗"
        label_text      = f"{d['label']} [{d['material']}] {rec_icon}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label_text,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return JSONResponse({
        "total_items":              total,
        "recyclable_count":         rec_count,
        "non_recyclable_count":     non_rec_count,
        "recyclability_rate":       f"{rec_rate}%",
        "avg_recyclability_score":  avg_score,
        "by_material":              by_material,
        "special_notes":            special_notes,
        "detections":               detections,
        "annotated_image":          b64(annotated),
        "original_image":           b64(image),
    })
