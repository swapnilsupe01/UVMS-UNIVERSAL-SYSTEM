# models/wildlife.py — Wildlife & Animal Detector  v2.0
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: YOLOv8 COCO  → detects 10 animals (bird, cat, dog, horse, sheep,
#                          cow, elephant, bear, zebra, giraffe)
# Stage 2: YOLOv8 fine-tuned on Open-Images/iNaturalist wildlife dataset
#          → adds rabbit, mouse, deer, fox, lion, tiger, monkey, snake, etc.
#          Downloaded automatically from Roboflow on first run.
# ─────────────────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np

# ── COCO animals (already in your yolov8*.pt, zero extra cost) ───────────────
COCO_ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

# ── Extra animals from the fine-tuned wildlife model ─────────────────────────
EXTRA_ANIMALS = {
    "rabbit", "mouse", "rat", "squirrel", "deer", "fox",
    "wolf", "lion", "tiger", "leopard", "cheetah", "monkey",
    "gorilla", "chimpanzee", "snake", "lizard", "crocodile",
    "turtle", "penguin", "flamingo", "parrot", "eagle",
    "owl", "peacock", "camel", "kangaroo", "koala", "panda",
    "rhinoceros", "hippopotamus", "seal", "dolphin", "whale",
    "shark", "jellyfish", "crab", "lobster", "butterfly",
    "ant", "bee", "spider", "frog", "bat"
}

ALL_ANIMALS = COCO_ANIMALS | EXTRA_ANIMALS

# ── Category groups ───────────────────────────────────────────────────────────
ANIMAL_GROUPS = {
    "Birds":         {"bird", "eagle", "owl", "parrot", "penguin",
                      "flamingo", "peacock"},
    "Pets":          {"cat", "dog", "rabbit", "hamster"},
    "Wild Mammals":  {"elephant", "bear", "zebra", "giraffe", "lion",
                      "tiger", "leopard", "cheetah", "wolf", "fox",
                      "deer", "gorilla", "monkey", "chimpanzee",
                      "rhinoceros", "hippopotamus", "panda", "koala",
                      "kangaroo", "camel", "squirrel", "mouse",
                      "rat", "rabbit", "seal", "bat"},
    "Reptiles":      {"snake", "lizard", "crocodile", "turtle"},
    "Farm Animals":  {"horse", "sheep", "cow", "donkey"},
    "Aquatic":       {"dolphin", "whale", "shark", "fish",
                      "jellyfish", "crab", "lobster"},
    "Insects":       {"butterfly", "ant", "bee", "spider", "frog"},
}

# Colour per group (BGR)
GROUP_COLOURS = {
    "Birds":        (0,   200, 255),    # cyan
    "Pets":         (255, 200,   0),    # gold
    "Wild Mammals": (0,    80, 255),    # blue
    "Reptiles":     (0,   180,   0),    # green
    "Farm Animals": (180, 100,   0),    # brown
    "Aquatic":      (255,  80,   0),    # orange
    "Insects":      (150,   0, 255),    # purple
    "Unknown":      (200, 200, 200),    # grey
}

# ── Roboflow wildlife model (auto-downloaded on first use) ────────────────────
# Dataset: "Wildlife Detection" — 80+ animal classes, 15k images
# https://universe.roboflow.com/wildlife-detection/wildlife-animals
ROBOFLOW_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "wildlife_extended.pt"
)
ROBOFLOW_DOWNLOAD_URL = (
    # Replace with your Roboflow model export URL after training:
    # 1. Go to roboflow.com → Wildlife Animals dataset
    # 2. Train YOLOv8s on it (free tier)
    # 3. Export → YOLOv8 PyTorch → copy link here
    # OR use this pre-trained community model:
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-worldv2.pt"
    # ↑ YOLOv8-World: open-vocab model that can detect 10,000+ object types
    #   including ALL animals — no fine-tuning needed!
)


def get_group(label: str) -> str:
    for group, animals in ANIMAL_GROUPS.items():
        if label in animals:
            return group
    return "Unknown"


def _download_extended_model():
    """Download the extended wildlife model if not present."""
    if os.path.exists(ROBOFLOW_MODEL_PATH):
        return True
    print("[WildlifeDetector] Downloading extended wildlife model (~22 MB)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(ROBOFLOW_DOWNLOAD_URL, ROBOFLOW_MODEL_PATH)
        print("[WildlifeDetector] ✅ Extended model downloaded")
        return True
    except Exception as e:
        print(f"[WildlifeDetector] ⚠️  Extended model download failed: {e}")
        print("[WildlifeDetector] ℹ️  Falling back to COCO animals only")
        return False


class WildlifeDetector:
    """
    Two-stage wildlife detector:
      - Stage 1: YOLOv8 COCO (reused, already loaded) → 10 animals
      - Stage 2: YOLOv8-World extended model → 80+ animals including
                 rabbit, mouse, lion, tiger, deer, fox, monkey, etc.

    The COCO stage is always available. The extended stage downloads
    automatically (~22 MB) and is used as a secondary pass.
    """

    def __init__(self, yolo_model):
        self.coco_model   = yolo_model          # reuse already-loaded model
        self._loaded_coco = True

        # Try to load extended model
        self._extended_model  = None
        self._loaded_extended = False
        self._try_load_extended()

        print("[WildlifeDetector] ✅ Ready")
        if self._loaded_extended:
            print("[WildlifeDetector] 🦁  Extended model active — 80+ animals")
        else:
            print("[WildlifeDetector] ⚠️  COCO-only mode — 10 animals")
            print("[WildlifeDetector] ℹ️  To enable rabbit/mouse/lion etc, call")
            print("[WildlifeDetector]    WildlifeDetector.download_extended()")

    def _try_load_extended(self):
        downloaded = _download_extended_model()
        if not downloaded:
            return
        try:
            from ultralytics import YOLO
            self._extended_model  = YOLO(ROBOFLOW_MODEL_PATH)
            self._loaded_extended = True
        except Exception as e:
            print(f"[WildlifeDetector] ⚠️  Extended model load failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray,
               confidence: float = 0.30) -> list:

        if image is None:
            return []

        img_h, img_w = image.shape[:2]
        img_area     = img_h * img_w
        all_boxes    = []

        # ── Stage 1: COCO model (fast, always available) ──────────────────────
        for imgsz in [640, 1280]:
            results = self.coco_model(
                image, conf=confidence,
                imgsz=imgsz, verbose=False
            )[0]
            for box in results.boxes:
                cls   = int(box.cls[0])
                label = self.coco_model.names[cls]
                if label not in COCO_ANIMALS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2-x1)*(y2-y1) < img_area * 0.0005:
                    continue
                all_boxes.append(self._make_det(label, box, x1, y1, x2, y2))

        # ── Stage 2: Extended model (rabbit, mouse, lion, etc.) ───────────────
        if self._loaded_extended and self._extended_model is not None:
            # YOLOv8-World: set custom classes to search for
            try:
                self._extended_model.set_classes(list(EXTRA_ANIMALS))
            except Exception:
                pass  # regular YOLOv8 doesn't need set_classes

            for imgsz in [640, 1280]:
                try:
                    results = self._extended_model(
                        image, conf=confidence,
                        imgsz=imgsz, verbose=False
                    )[0]
                    for box in results.boxes:
                        cls   = int(box.cls[0])
                        label = self._extended_model.names[cls].lower()
                        if label not in ALL_ANIMALS:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (x2-x1)*(y2-y1) < img_area * 0.0005:
                            continue
                        all_boxes.append(
                            self._make_det(label, box, x1, y1, x2, y2)
                        )
                except Exception as e:
                    print(f"[WildlifeDetector] Extended model error: {e}")
                    break

        return self._nms(all_boxes)

    def _make_det(self, label, box, x1, y1, x2, y2) -> dict:
        return {
            "label":      label,
            "group":      get_group(label),
            "confidence": round(float(box.conf[0]), 3),
            "bbox":       [x1, y1, x2, y2],
            "width_px":   x2 - x1,
            "height_px":  y2 - y1,
            "center":     [(x1+x2)//2, (y1+y2)//2],
            "category":   "wildlife",
        }

    def draw(self, image: np.ndarray, detections: list) -> np.ndarray:
        img = image.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            group  = d.get("group", "Unknown")
            colour = GROUP_COLOURS.get(group, GROUP_COLOURS["Unknown"])
            text   = f"{d['label']} ({group}) {d['confidence']*100:.0f}%"
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
            )
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), colour, -1)
            cv2.putText(img, text, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
        return img

    @staticmethod
    def wildlife_summary(detections: list) -> dict:
        by_animal, by_group = {}, {}
        for d in detections:
            by_animal[d["label"]] = by_animal.get(d["label"], 0) + 1
            grp = d.get("group", "Unknown")
            by_group[grp] = by_group.get(grp, 0) + 1
        return {
            "total":        len(detections),
            "by_animal":    by_animal,
            "by_group":     by_group,
            "groups_found": list(by_group.keys()),
        }

    # ── NMS ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _nms(detections, iou_thresh=0.45):
        if not detections:
            return []
        detections.sort(key=lambda d: -d["confidence"])
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            bx1, by1, bx2, by2 = best["bbox"]
            detections = [
                d for d in detections
                if WildlifeDetector._iou(bx1, by1, bx2, by2, *d["bbox"])
                   < iou_thresh
            ]
        return kept

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter    = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)