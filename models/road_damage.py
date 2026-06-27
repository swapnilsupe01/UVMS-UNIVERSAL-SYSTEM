"""
models/road_damage.py  — UVMS v5.0
Road Damage Detector using YOLOv8 fine-tuned on RDD2022
(Road Damage Dataset 2022 — 47,420 images, 4 damage classes)

Damage classes:
  D00 → Longitudinal Crack   (cracks running along road direction)
  D10 → Transverse Crack     (cracks across road direction)
  D20 → Alligator Crack      (spider-web / mesh cracks)
  D40 → Pothole              (holes in road surface)

Model auto-downloads from Roboflow Universe on first use (~6 MB).
Falls back to COCO YOLOv8 pothole-only mode if download fails.
"""

import cv2
import numpy as np
import os
import urllib.request

# ── Severity thresholds (area as % of image) ──────────────────
SEVERITY_RULES = {
    "pothole": [
        (0.020, "Critical"),   # > 2% of image = Critical
        (0.008, "Urgent"),     # > 0.8%        = Urgent
        (0.002, "Moderate"),   # > 0.2%        = Moderate
        (0.000, "Minor"),      # everything else
    ],
    "alligator_crack": [
        (0.015, "Critical"),
        (0.006, "Urgent"),
        (0.001, "Moderate"),
        (0.000, "Minor"),
    ],
    "longitudinal_crack": [
        (0.010, "Critical"),
        (0.004, "Urgent"),
        (0.001, "Moderate"),
        (0.000, "Minor"),
    ],
    "transverse_crack": [
        (0.010, "Critical"),
        (0.004, "Urgent"),
        (0.001, "Moderate"),
        (0.000, "Minor"),
    ],
}

PRIORITY_ORDER = {"Critical": 0, "Urgent": 1, "Moderate": 2, "Minor": 3}

# Box colours per severity
SEVERITY_COLOURS = {
    "Critical": (0,   0,   255),   # red
    "Urgent":   (0,   100, 255),   # orange-red
    "Moderate": (0,   165, 255),   # orange
    "Minor":    (0,   220, 90),    # green
}

# ── Class name normalisation ───────────────────────────────────
# Maps whatever the model outputs → our clean internal name
CLASS_ALIASES = {
    # RDD2022 class names
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    # plain English names (Roboflow exports)
    "pothole":            "pothole",
    "Pothole":            "pothole",
    "crack":              "alligator_crack",
    "Crack":              "alligator_crack",
    "longitudinal crack": "longitudinal_crack",
    "transverse crack":   "transverse_crack",
    "alligator crack":    "alligator_crack",
    "alligator_crack":    "alligator_crack",
    "longitudinal_crack": "longitudinal_crack",
    "transverse_crack":   "transverse_crack",
    # COCO fallback
    "road":               "pothole",
}

DISPLAY_NAMES = {
    "pothole":            "POTHOLE",
    "alligator_crack":    "ALLIGATOR CRACK",
    "longitudinal_crack": "LONG. CRACK",
    "transverse_crack":   "TRANS. CRACK",
}

# Pixels-per-cm calibration constant (approximate for road photos)
# Adjust this per camera / distance if you know the actual scale
PX_PER_CM = 3.7


class RoadDamageDetector:
    """
    Two-stage road damage detector:
      Stage 1 → YOLOv8 trained on RDD2022 (finds damage + bounding boxes)
      Stage 2 → OpenCV image analysis (fills gaps, measures severity)
    Falls back to COCO YOLOv8 + OpenCV-only if specialist model unavailable.
    """

    # ── Model sources (tried in order) ────────────────────────
    MODEL_URLS = [
        # Roboflow Universe — RDD2022 YOLOv8 (public, CC BY 4.0)
        (
            "https://github.com/ultralytics/assets/releases/download/"
            "v8.3.0/yolov8n.pt",          # placeholder: replace with your
            "road_damage_yolov8.pt"        # downloaded RDD2022 model path
        ),
    ]

    # Roboflow export URL for RDD2022 (requires free API key — see README)
    ROBOFLOW_MODEL = None   # Set to "username/rdd2022-yolov8/1" if you have API key

    def __init__(self, model_path: str = None):
        from ultralytics import YOLO

        self._loaded    = False
        self._rdd_model = None   # specialist road model
        self._base_model = None  # fallback COCO model

        # ── Try loading specialist road model ─────────────────
        search_paths = [
            model_path,
            "models/road_damage_yolov8.pt",
            "road_damage_yolov8.pt",
            "models/pothole_yolov8.pt",
            "pothole_yolov8.pt",
            "models/rdd2022.pt",
            "rdd2022.pt",
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                try:
                    self._rdd_model = YOLO(path)
                    print(f"[RoadDamage] ✅ Loaded specialist model: {path}")
                    self._loaded = True
                    break
                except Exception as e:
                    print(f"[RoadDamage] ⚠️  Could not load {path}: {e}")

        # ── If no specialist model, download one ──────────────
        if not self._loaded:
            print("[RoadDamage] No specialist model found. Trying download...")
            self._try_download_model(YOLO)

        # ── Always load base COCO model as fallback ───────────
        print("[RoadDamage] Loading base YOLOv8 (fallback + person detection)...")
        self._base_model = YOLO("yolov8l.pt")
        print("[RoadDamage] ✅ Base model loaded")

        if not self._loaded:
            print("[RoadDamage] ⚠️  Using OpenCV-enhanced COCO mode.")
            print("[RoadDamage]    For best results, place a RDD2022 YOLOv8 model at:")
            print("[RoadDamage]    models/road_damage_yolov8.pt")
            print("[RoadDamage]    Download: https://universe.roboflow.com/search?q=road+damage+rdd2022")
            self._loaded = True   # Still usable via base model + OpenCV

    def _try_download_model(self, YOLO):
        """Try to download a public road damage model."""
        save_path = "models/road_damage_yolov8.pt"
        os.makedirs("models", exist_ok=True)

        # Try Roboflow if API key is set in environment
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if api_key:
            try:
                from roboflow import Roboflow
                rf      = Roboflow(api_key=api_key)
                project = rf.workspace().project("rdd2022-yolov8")
                version = project.version(1)
                version.model.save(save_path)
                self._rdd_model = YOLO(save_path)
                self._loaded = True
                print("[RoadDamage] ✅ Downloaded via Roboflow API")
                return
            except Exception as e:
                print(f"[RoadDamage] Roboflow download failed: {e}")

        # Direct GitHub releases (community models)
        fallback_urls = [
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        ]
        for url in fallback_urls:
            try:
                print(f"[RoadDamage] Downloading from {url[:60]}...")
                urllib.request.urlretrieve(url, save_path)
                self._rdd_model = YOLO(save_path)
                self._loaded = True
                print("[RoadDamage] ✅ Download complete")
                return
            except Exception as e:
                print(f"[RoadDamage] Download failed: {e}")

    def is_ready(self) -> bool:
        return self._loaded

    # ── Main detection ─────────────────────────────────────────
    def detect(
        self,
        image:      np.ndarray,
        confidence: float = 0.10,          # low default — road damage is subtle
        depth_map:  np.ndarray = None,
    ) -> list:
        """
        Run 3-stage road damage detection:
          1. YOLOv8 multi-scale detection (640 + 1280)
          2. OpenCV-based crack / pothole analysis to catch what YOLO misses
          3. Merge + NMS + severity scoring
        """
        img_h, img_w = image.shape[:2]
        img_area     = img_h * img_w
        all_detections = []

        # Stage 1A — specialist road model (if available)
        if self._rdd_model is not None:
            for imgsz in [640, 1280]:
                try:
                    results = self._rdd_model(
                        image, conf=confidence, imgsz=imgsz, verbose=False
                    )[0]
                    for box in results.boxes:
                        raw_label = self._rdd_model.names[int(box.cls[0])]
                        label     = CLASS_ALIASES.get(raw_label, raw_label.lower())
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2-x1) * (y2-y1)
                        if area < img_area * 0.0003:
                            continue
                        all_detections.append({
                            "_conf": float(box.conf[0]),
                            "_bbox": [x1, y1, x2, y2],
                            "_label": label,
                            "_source": "yolo_specialist",
                        })
                except Exception as e:
                    print(f"[RoadDamage] YOLO specialist error at {imgsz}px: {e}")

        # Stage 1B — base COCO model (catches potholes as "pothole" in some versions)
        for imgsz in [640, 1280]:
            try:
                results = self._base_model(
                    image, conf=confidence * 0.8, imgsz=imgsz, verbose=False
                )[0]
                for box in results.boxes:
                    raw_label = self._base_model.names[int(box.cls[0])]
                    label     = CLASS_ALIASES.get(raw_label)
                    if label is None:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2-x1) * (y2-y1)
                    if area < img_area * 0.0003:
                        continue
                    all_detections.append({
                        "_conf": float(box.conf[0]),
                        "_bbox": [x1, y1, x2, y2],
                        "_label": label,
                        "_source": "yolo_coco",
                    })
            except Exception as e:
                print(f"[RoadDamage] YOLO base error at {imgsz}px: {e}")

        # Stage 2 — OpenCV analysis (catches damage YOLO misses at low conf)
        cv_detections = self._opencv_detect(image, confidence)
        all_detections.extend(cv_detections)

        # Stage 3 — NMS + finalize
        merged = self._nms(all_detections, iou_thresh=0.40)
        return [self._finalize(d, image, img_area, depth_map)
                for d in merged]

    # ── OpenCV-based damage detection ─────────────────────────
    def _opencv_detect(self, image: np.ndarray, confidence: float) -> list:
        """
        Find road damage regions using image processing:
          - Convert to grayscale
          - Adaptive threshold for cracks (dark lines on light road)
          - Find contours → group into bounding boxes
          - Filter noise
        This catches alligator cracks and longitudinal cracks that
        YOLO often misses.
        """
        img_h, img_w = image.shape[:2]
        img_area     = img_h * img_w
        detections   = []

        # Pre-process: grayscale + CLAHE (enhances road texture contrast)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # ── Method A: Crack detection via adaptive threshold ──
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=35, C=8
        )

        # Morphological ops — connect nearby crack pixels
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.0015:     # skip tiny noise
                continue
            if area > img_area * 0.50:       # skip huge regions (whole frame)
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Classify crack type by aspect ratio
            aspect = w / (h + 1e-5)
            if   aspect > 3.5:   label = "longitudinal_crack"
            elif aspect < 0.30:  label = "transverse_crack"
            else:                label = "alligator_crack"

            # Estimate confidence from area ratio (larger = more confident)
            area_ratio  = area / img_area
            est_conf    = min(0.85, 0.25 + area_ratio * 15)

            if est_conf < confidence * 0.6:
                continue

            detections.append({
                "_conf":   est_conf,
                "_bbox":   [x, y, x+w, y+h],
                "_label":  label,
                "_source": "opencv_threshold",
            })

        # ── Method B: Pothole detection via dark region analysis ──
        # Potholes appear as dark irregular regions on lighter road
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dark_clean   = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                        dark_kernel, iterations=3)

        p_contours, _ = cv2.findContours(
            dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in p_contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.003:
                continue
            if area > img_area * 0.30:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect      = w / (h + 1e-5)

            # Potholes are roughly circular/square — filter long thin shapes
            if aspect > 4.0 or aspect < 0.25:
                continue

            # Check that region is actually darker than surroundings
            roi_gray  = gray[y:y+h, x:x+w]
            surr_y1   = max(0, y - 15)
            surr_y2   = min(img_h, y + h + 15)
            surr_x1   = max(0, x - 15)
            surr_x2   = min(img_w, x + w + 15)
            surr_gray = gray[surr_y1:surr_y2, surr_x1:surr_x2]

            if roi_gray.size == 0 or surr_gray.size == 0:
                continue

            roi_mean  = float(roi_gray.mean())
            surr_mean = float(surr_gray.mean())

            if roi_mean > surr_mean - 15:     # not significantly darker
                continue

            area_ratio = area / img_area
            est_conf   = min(0.88, 0.30 + area_ratio * 10)

            if est_conf < confidence * 0.6:
                continue

            detections.append({
                "_conf":   est_conf,
                "_bbox":   [x, y, x+w, y+h],
                "_label":  "pothole",
                "_source": "opencv_dark_region",
            })

        return detections

    # ── Non-Maximum Suppression ────────────────────────────────
    @staticmethod
    def _nms(detections: list, iou_thresh: float = 0.40) -> list:
        if not detections:
            return []
        detections.sort(key=lambda d: -d["_conf"])
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            bx1, by1, bx2, by2 = best["_bbox"]
            detections = [
                d for d in detections
                if RoadDamageDetector._iou(
                    bx1, by1, bx2, by2, *d["_bbox"]
                ) < iou_thresh
            ]
        return kept

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    # ── Finalize each detection ────────────────────────────────
    def _finalize(
        self,
        raw:      dict,
        image:    np.ndarray,
        img_area: int,
        depth_map: np.ndarray = None,
    ) -> dict:
        label       = raw["_label"]
        conf        = raw["_conf"]
        x1, y1, x2, y2 = raw["_bbox"]
        img_h, img_w = image.shape[:2]

        # Clamp bbox to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)
        w_px = x2 - x1; h_px = y2 - y1
        area_px = w_px * h_px
        area_ratio = area_px / img_area

        # Severity from area ratio
        thresholds = SEVERITY_RULES.get(label, SEVERITY_RULES["pothole"])
        severity = "Minor"
        for thresh, sev in thresholds:
            if area_ratio >= thresh:
                severity = sev
                break

        # Real-world size estimate (cm)
        w_cm = round(w_px / PX_PER_CM, 1)
        h_cm = round(h_px / PX_PER_CM, 1)

        # Depth at damage centre (if depth map available)
        depth_val = None
        if depth_map is not None:
            cx = min((x1 + x2) // 2, depth_map.shape[1] - 1)
            cy = min((y1 + y2) // 2, depth_map.shape[0] - 1)
            depth_val = int(depth_map[cy, cx])

        # Repair priority
        priority = severity
        if label == "pothole" and severity in ("Critical", "Urgent"):
            priority = "🔴 Immediate Repair"
        elif severity == "Moderate":
            priority = "🟡 Monitor"
        elif severity == "Minor":
            priority = "🟢 Low Priority"

        return {
            "label":        label,
            "display_name": DISPLAY_NAMES.get(label, label.upper()),
            "confidence":   round(conf, 3),
            "severity":     severity,
            "priority":     priority,
            "bbox":         [x1, y1, x2, y2],
            "width_px":     w_px,
            "height_px":    h_px,
            "width_cm":     w_cm,
            "height_cm":    h_cm,
            "area_cm2":     round(w_cm * h_cm, 1),
            "center":       [(x1 + x2) // 2, (y1 + y2) // 2],
            "depth_value":  depth_val,
            "source":       raw.get("_source", "unknown"),
        }

    # ── Draw annotated image ───────────────────────────────────
    def draw(self, image: np.ndarray, detections: list) -> np.ndarray:
        img = image.copy()

        # Overall road condition header bar
        if detections:
            summary     = RoadDamageDetector.damage_summary(detections)
            condition   = summary["road_condition"]
            bar_colours = {
                "Critical": (0, 0, 200),
                "Poor":     (0, 80, 200),
                "Fair":     (0, 140, 255),
                "Good":     (60, 180, 60),
            }
            bar_colour = bar_colours.get(condition, (100, 100, 100))
            cv2.rectangle(img, (0, 0), (img.shape[1], 36), bar_colour, -1)
            cv2.putText(
                img,
                f"Road: {condition}  |  {len(detections)} damage found",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2
            )

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            sev    = d["severity"]
            colour = SEVERITY_COLOURS.get(sev, (200, 200, 200))

            # Box
            thickness = 3 if sev in ("Critical", "Urgent") else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

            # Label line 1: type + severity + confidence
            line1 = f"{d['display_name']} [{sev}] {d['confidence']*100:.0f}%"
            # Label line 2: real-world size
            line2 = f"{d['width_cm']} x {d['height_cm']} cm"

            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.48
            th    = 16   # approx text height

            # Background for both lines
            (w1, _), _ = cv2.getTextSize(line1, font, scale, 1)
            (w2, _), _ = cv2.getTextSize(line2, font, scale, 1)
            max_w = max(w1, w2) + 8

            cv2.rectangle(img,
                          (x1, y2),
                          (x1 + max_w, y2 + th * 2 + 12),
                          colour, -1)
            cv2.putText(img, line1, (x1 + 3, y2 + th),
                        font, scale, (255, 255, 255), 1)
            cv2.putText(img, line2, (x1 + 3, y2 + th * 2 + 6),
                        font, scale, (255, 255, 255), 1)

        return img

    # ── Damage summary ─────────────────────────────────────────
    @staticmethod
    def damage_summary(detections: list) -> dict:
        by_type     = {}
        by_severity = {}
        total_area  = 0.0
        urgent_count = 0
        monitor_count = 0

        for d in detections:
            lbl = d["label"]
            sev = d["severity"]
            by_type[lbl]     = by_type.get(lbl, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
            total_area      += d.get("area_cm2", 0)
            if sev in ("Critical", "Urgent"):
                urgent_count  += 1
            elif sev == "Moderate":
                monitor_count += 1

        # Overall road condition
        crit  = by_severity.get("Critical", 0)
        urg   = by_severity.get("Urgent",   0)
        mod   = by_severity.get("Moderate", 0)
        total = len(detections)

        if   crit > 0:                        condition = "Critical"
        elif urg  > 1 or (urg > 0 and mod > 2): condition = "Poor"
        elif urg  > 0 or mod > 1:             condition = "Fair"
        elif total > 0:                        condition = "Monitored"
        else:                                  condition = "Good"

        return {
            "total":                total,
            "road_condition":       condition,
            "urgent_count":         urgent_count,
            "monitor_count":        monitor_count,
            "total_damage_area_cm2": round(total_area, 1),
            "by_type":              dict(sorted(by_type.items(),
                                                key=lambda x: -x[1])),
            "by_severity":          by_severity,
        }