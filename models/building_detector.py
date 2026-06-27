
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
 
BUILDING_LABELS = {
    "Building", "House", "Skyscraper", "Tower",
    "Office building", "Apartment building", "Castle",
    "Church", "Lighthouse", "Barn", "Factory", "Bridge",
}
COLOUR = (255, 140, 0)   # orange
 
 
class BuildingDetector:
    """
    Uses Google Open Images SSD MobileNetV2 (TF Hub) for building detection.
    Compatible with any TF version — no version pin required.
    All shape bugs from v3 are fixed with np.atleast_1d / flatten.
    """
 
    def __init__(self):
        print("[BuildingDetector] Loading TF Hub Open Images SSD model...")
        self._loaded = False
        try:
            self.model = hub.load(
                "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
            )
            self.detector_fn = self.model.signatures["default"]
            print("[BuildingDetector] ✅ Model loaded")
            self._loaded = True
        except Exception as e:
            print(f"[BuildingDetector] ❌ Load failed: {e}")
 
    def detect(self, image: "np.ndarray", confidence: float = 0.40) -> list:
        if not self._loaded:
            return []
 
        img_h, img_w = image.shape[:2]
        img_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)
        img_tensor = img_tensor[tf.newaxis, ...]           # [1, H, W, 3]
 
        result = self.detector_fn(img_tensor)
 
        # ── BUG FIX from v3: use flatten + atleast_1d so scalars become arrays ──
        scores      = np.atleast_1d(result["detection_scores"].numpy().flatten())
        raw_boxes   = result["detection_boxes"].numpy()
        boxes       = raw_boxes[0] if raw_boxes.ndim == 3 else raw_boxes
        boxes       = np.atleast_2d(boxes)
 
        raw_ents    = result["detection_class_entities"].numpy()
        class_names = raw_ents[0] if raw_ents.ndim == 2 else raw_ents
        class_names = np.atleast_1d(class_names)
 
        detections = []
        for i in range(len(scores)):
            if float(scores[i]) < confidence:
                continue
 
            label = class_names[i].decode("utf-8") if i < len(class_names) else "Building"
            if label not in BUILDING_LABELS:
                continue
 
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * img_w);  y1 = int(ymin * img_h)
            x2 = int(xmax * img_w);  y2 = int(ymax * img_h)
            w,  h = x2 - x1, y2 - y1
 
            if w * h < img_h * img_w * 0.005:
                continue
 
            detections.append({
                "label":      label,
                "confidence": round(float(scores[i]), 3),
                "bbox":       [x1, y1, x2, y2],
                "width_px":   w,
                "height_px":  h,
                "center":     [(x1+x2)//2, (y1+y2)//2],
                "floors_est": max(1, round(h / 35)),
                "category":   "building",
            })
 
        return detections
 
    def draw(self, image, detections):
        img = image.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            floors = d.get("floors_est", "?")
            text   = f"{d['label']} {d['confidence']*100:.0f}% (~{floors} floors)"
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y2), (x1+tw+8, y2+th+10), COLOUR, -1)
            cv2.putText(img, text, (x1+4, y2+th+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return img
