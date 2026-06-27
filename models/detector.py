from ultralytics import YOLO
import cv2, numpy as np, os

# ── Class maps ────────────────────────────────────────────────
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "van", "train", "boat", "airplane"
}

# FIX: Only REAL COCO class names. "sofa","table","desk","wardrobe",
# "cabinet","bookshelf","shelf" do NOT exist in COCO — YOLO never
# outputs them, so they silently swallowed every detection.
# Full COCO furniture-ish set:
FURNITURE_CLASSES = {
    "chair",        # ✅ COCO
    "couch",        # ✅ COCO  (was also listed as "sofa" — wrong name)
    "bed",          # ✅ COCO
    "dining table", # ✅ COCO  (was also listed as "table" — wrong name)
    "toilet",       # ✅ COCO
    "tv",           # ✅ COCO  (added — common in room photos)
    "laptop",       # ✅ COCO  (added)
    "microwave",    # ✅ COCO  (added)
    "oven",         # ✅ COCO  (added)
    "refrigerator", # ✅ COCO  (added)
    "sink",         # ✅ COCO  (added)
    "clock",        # ✅ COCO  (added)
    "vase",         # ✅ COCO  (added)
    "potted plant", # ✅ COCO  (added)
    "book",         # ✅ COCO  (added)
    "remote",       # ✅ COCO  (added — TV remote)
    "mouse",        # ✅ COCO  (computer mouse)
    "keyboard",     # ✅ COCO
}

# Waste: each item has material type, recyclability, score, optional note
WASTE_CLASSES = {
    "bottle":     {"material": "Plastic",        "recyclable": True,  "score": 0.95},
    "cup":        {"material": "Paper/Plastic",   "recyclable": True,  "score": 0.70},
    "wine glass": {"material": "Glass",           "recyclable": True,  "score": 0.90},
    "bowl":       {"material": "Ceramic",         "recyclable": False, "score": 0.20},
    "can":        {"material": "Metal",           "recyclable": True,  "score": 0.98},
    "fork":       {"material": "Metal",           "recyclable": True,  "score": 0.85},
    "knife":      {"material": "Metal",           "recyclable": True,  "score": 0.85},
    "spoon":      {"material": "Metal",           "recyclable": True,  "score": 0.85},
    "scissors":   {"material": "Metal",           "recyclable": True,  "score": 0.80},
    "vase":       {"material": "Glass/Ceramic",   "recyclable": True,  "score": 0.60},
    "cell phone": {"material": "E-Waste",         "recyclable": False, "score": 0.0,
                   "note": "⚠️ Special E-Waste Disposal Required"},
    "laptop":     {"material": "E-Waste",         "recyclable": False, "score": 0.0,
                   "note": "⚠️ Special E-Waste Disposal Required"},
    "keyboard":   {"material": "E-Waste",         "recyclable": False, "score": 0.0,
                   "note": "⚠️ Special E-Waste Disposal Required"},
    "remote":     {"material": "E-Waste",         "recyclable": False, "score": 0.0,
                   "note": "⚠️ Special E-Waste Disposal Required"},
    "book":       {"material": "Paper",           "recyclable": True,  "score": 0.95},
    "backpack":   {"material": "Fabric",          "recyclable": False, "score": 0.10},
    "handbag":    {"material": "Fabric",          "recyclable": False, "score": 0.10},
    "suitcase":   {"material": "Plastic/Fabric",  "recyclable": False, "score": 0.15},
    "umbrella":   {"material": "Fabric/Metal",    "recyclable": False, "score": 0.20},
    "frisbee":    {"material": "Plastic",         "recyclable": True,  "score": 0.70},
    "banana":     {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
    "apple":      {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
    "orange":     {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
    "pizza":      {"material": "Organic/Paper",   "recyclable": False, "score": 0.0,
                   "note": "❌ Contaminated - Cannot Recycle"},
    "sandwich":   {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
    "hot dog":    {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
    "cake":       {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
    "donut":      {"material": "Organic",         "recyclable": False, "score": 0.0,
                   "note": "🌱 Compost"},
}

COLOURS = {
    "vehicle":   (0,   230, 180),
    "furniture": (255, 165,   0),
    "waste":     (50,  205,  50),
    "weapon":    (0,     0, 255),
    "building":  (255, 140,   0),
    "default":   (200, 200, 200),
}


class MultiDetector:
    def __init__(self, model_size="l"):
        self.model = YOLO(f"yolov8{model_size}.pt")
        print(f"[MultiDetector] ✅ Loaded yolov8{model_size}.pt")
        self._weapon_model      = None
        self._weapon_model_path = "yolov8_weapon.pt"

    def _load_weapon_model(self):
        if self._weapon_model is not None:
            return
        if not os.path.exists(self._weapon_model_path):
            print("[WEAPON] Downloading specialist weapon model...")
            import urllib.request
            url = (
                "https://github.com/nicehorse06/yolov8-weapon-detection/"
                "releases/download/v1.0/best.pt"
            )
            try:
                urllib.request.urlretrieve(url, self._weapon_model_path)
                print("[WEAPON] ✅ Download complete")
            except Exception as e:
                print(f"[WEAPON] ⚠️ Download failed: {e}")
                self._weapon_model = self.model
                return
        self._weapon_model = YOLO(self._weapon_model_path)
        print("[WEAPON] ✅ Weapon model loaded")

    def _multi_scale_detect(self, image, confidence, class_filter=None):
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w
        all_boxes = []
        for imgsz in [640, 1280]:
            results = self.model(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
            for box in results.boxes:
                cls   = int(box.cls[0])
                label = self.model.names[cls]
                if class_filter and label not in class_filter:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < img_area * 0.0005 or area > img_area * 0.95:
                    continue
                all_boxes.append({
                    "label":      label,
                    "confidence": float(box.conf[0]),
                    "bbox":       [x1, y1, x2, y2],
                    "width_px":   x2 - x1,
                    "height_px":  y2 - y1,
                    "center":     [(x1 + x2) // 2, (y1 + y2) // 2],
                })
        return self._nms(all_boxes, iou_thresh=0.45)

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
                if MultiDetector._iou(bx1, by1, bx2, by2, *d["bbox"]) < iou_thresh
            ]
        return kept

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def detect(self, image, confidence=0.40, mode="vehicle"):
        if   mode == "vehicle":   return self._detect_vehicle(image, confidence)
        elif mode == "furniture": return self._detect_furniture(image, confidence)
        elif mode == "waste":     return self._detect_waste(image, confidence)
        elif mode == "weapon":    return self._detect_weapon(image, confidence)
        return []

    def _detect_vehicle(self, image, confidence):
        detections = self._multi_scale_detect(image, confidence, VEHICLE_CLASSES)
        for d in detections:
            d["mode"]       = "vehicle"
            d["confidence"] = round(d["confidence"], 3)
        return detections

    def _detect_furniture(self, image, confidence):
        # FIX: use _multi_scale_detect (same as vehicle) for better coverage
        detections = self._multi_scale_detect(image, confidence, FURNITURE_CLASSES)
        for d in detections:
            d["mode"]       = "furniture"
            d["confidence"] = round(d["confidence"], 3)
        return detections

    def _detect_waste(self, image, confidence):
        img_area = image.shape[0] * image.shape[1]
        results  = self.model(image, conf=confidence, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = self.model.names[cls]
            if label not in WASTE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2-x1)*(y2-y1) < img_area * 0.001:
                continue
            info = WASTE_CLASSES[label]
            det = {
                "label":               label,
                "confidence":          round(float(box.conf[0]), 3),
                "bbox":                [x1, y1, x2, y2],
                "width_px":            x2 - x1,
                "height_px":           y2 - y1,
                "center":              [(x1+x2)//2, (y1+y2)//2],
                "mode":                "waste",
                "material":            info["material"],
                "recyclable":          info["recyclable"],
                "recyclability_score": info["score"],
                "verdict":             "✅ Recyclable" if info["recyclable"] else "❌ Not Recyclable",
            }
            if "note" in info:
                det["special_note"] = info["note"]
            detections.append(det)
        return detections

    def _detect_weapon(self, image, confidence):
        img_area = image.shape[0] * image.shape[1]
        person_results = self.model(image, conf=0.35, verbose=False)[0]
        persons = []
        for box in person_results.boxes:
            if self.model.names[int(box.cls[0])] == "person":
                persons.append(list(map(int, box.xyxy[0])))
        self._load_weapon_model()
        weapon_results = self._weapon_model(image, conf=confidence, verbose=False)[0]
        detections = []
        for box in weapon_results.boxes:
            cls   = int(box.cls[0])
            label = self._weapon_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2-x1)*(y2-y1) < img_area * 0.001:
                continue
            wx, wy = (x1+x2)//2, (y1+y2)//2
            nearest_person = None
            min_dist = float("inf")
            for p in persons:
                px, py = (p[0]+p[2])//2, (p[1]+p[3])//2
                dist = ((wx-px)**2 + (wy-py)**2) ** 0.5
                if dist < min_dist:
                    min_dist, nearest_person = dist, p
            detections.append({
                "label":         label,
                "confidence":    round(float(box.conf[0]), 3),
                "bbox":          [x1, y1, x2, y2],
                "width_px":      x2 - x1,
                "height_px":     y2 - y1,
                "center":        [wx, wy],
                "mode":          "weapon",
                "alert":         "⚠️ WEAPON DETECTED",
                "person_nearby": nearest_person is not None,
                "person_bbox":   nearest_person,
                "threat_level":  "🔴 HIGH" if nearest_person else "🟡 MEDIUM",
            })
        return detections

    @staticmethod
    def waste_summary(detections):
        by_item, by_material = {}, {}
        recyclable_count = 0
        special_notes    = []
        total_score      = 0.0
        for d in detections:
            lbl = d["label"]
            mat = d.get("material", "Unknown")
            by_item[lbl]     = by_item.get(lbl, 0) + 1
            by_material[mat] = by_material.get(mat, 0) + 1
            if d.get("recyclable"):
                recyclable_count += 1
            total_score += d.get("recyclability_score", 0)
            note = d.get("special_note")
            if note and note not in special_notes:
                special_notes.append(note)
        total = len(detections)
        return {
            "total":                total,
            "recyclable_count":     recyclable_count,
            "non_recyclable_count": total - recyclable_count,
            "recyclability_rate":   f"{(recyclable_count/total*100):.0f}%" if total else "0%",
            "avg_recyclability_score": round(total_score / total, 2) if total else 0,
            "by_item":     dict(sorted(by_item.items(),     key=lambda x: -x[1])),
            "by_material": dict(sorted(by_material.items(), key=lambda x: -x[1])),
            "special_notes": special_notes,
        }

    def draw(self, image, detections, mode="vehicle"):
        img   = image.copy()
        color = COLOURS.get(mode, COLOURS["default"])
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            pct = d["confidence"] * 100
            if mode == "vehicle":
                gc   = d.get("ground_clearance_cm", "")
                text = f"{d['label']} {pct:.0f}%" + (f" | {gc}cm" if gc else "")
            elif mode == "waste":
                sym  = "♻" if d.get("recyclable") else "✗"
                text = f"{sym} {d['label']} [{d.get('material','')}]"
                color = (50, 205, 50) if d.get("recyclable") else (0, 100, 255)
            elif mode == "weapon":
                color = (0, 0, 255)
                text  = f"⚠ {d['label']} {pct:.0f}% {d.get('threat_level','')}"
                if d.get("person_bbox"):
                    px1, py1, px2, py2 = d["person_bbox"]
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 140, 255), 3)
                    cv2.putText(img, "ARMED PERSON", (px1 + 4, py1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
            else:
                text = f"{d['label']} {pct:.0f}%"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(img, text, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
        return img