import cv2
import numpy as np
import base64
from fastapi import HTTPException


def read_img(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            import io
            pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def require_image(file_bytes: bytes):
    img = read_img(file_bytes)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot read image. Supported: JPG, PNG, WEBP, AVIF, BMP, HEIC"
        )
    return img


def b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()
