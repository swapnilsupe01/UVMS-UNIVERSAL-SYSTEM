# """
# main.py — FIXED
# -----------------
# Bug 1 fixed: app was created TWICE. The second `app = FastAPI(...)` wiped
#              the first instance including its middleware. Now created once.

# Bug 2 fixed: Models are now stored on app.state and loaded via a startup
#              event — so routers can access the SAME singleton via
#              request.app.state.detector instead of creating new instances.
# """

# import traceback
# import cv2
# import numpy as np
# import base64
# import torch

# from fastapi import FastAPI, File, UploadFile, Query, HTTPException,Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, FileResponse
# from contextlib import asynccontextmanager

# from models.detector          import MultiDetector
# from models.building_detector import BuildingDetector
# from models.depth             import DepthEstimator


# # ── Lifespan: load models once at startup, store on app.state ─────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("🚀 Loading AI models...")
#     try:
#         app.state.detector         = MultiDetector(model_size="l")
#         print("✅ detector loaded")
#         app.state.building_detector = BuildingDetector()
#         print("✅ building detector loaded")
#         app.state.depth_est        = DepthEstimator()
#         print("✅ depth estimator loaded")
#     except Exception:
#         print("❌ Model load failed:")
#         traceback.print_exc()
#         raise
#     yield
#     # (shutdown cleanup goes here if needed)


# # ── Single app instance ────────────────────────────────────────────────────────
# app = FastAPI(
#     title="UVMS API v4.0",
#     version="4.0.0",
#     lifespan=lifespan,
#     description="""
# ## Urban Vision Management System

# | Endpoint | Purpose |
# |---|---|
# | **POST /furniture/** | Detect interior furniture |
# | **POST /building/** | Count windows & estimate floors |
# | **POST /waste/** | Identify waste material & recyclability |
# """,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── Register routers AFTER app is created ─────────────────────────────────────
# from routes.furniture import router as furniture_router
# from routes.building  import router as building_router
# from routes.waste     import router as waste_router

# app.include_router(furniture_router)
# app.include_router(building_router)
# app.include_router(waste_router)


# # ── Helpers ───────────────────────────────────────────────────────────────────
# def b64(img):
#     _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     return base64.b64encode(buf).decode()


# def read_img(file_bytes: bytes):
#     arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if img is None:
#         try:
#             from PIL import Image
#             import io
#             pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#             img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
#         except Exception:
#             return None
#     return img


# def require_image(file_bytes: bytes):
#     img = read_img(file_bytes)
#     if img is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot read image. Supported: JPG, PNG, WEBP, AVIF, BMP, HEIC"
#         )
#     return img


# def add_ground_clearance(detections, image, depth_est):
#     if not detections:
#         return detections, None
#     depth_map = depth_est.estimate(image)
#     for d in detections:
#         x1, y1, x2, y2 = d["bbox"]
#         cx       = (x1 + x2) // 2
#         car_y    = min(y2,      depth_map.shape[0] - 1)
#         ground_y = min(y2 + 15, depth_map.shape[0] - 1)
#         d["ground_clearance_cm"] = round(
#             abs(int(depth_map[ground_y, cx]) - int(depth_map[car_y, cx])) * 1.5, 1
#         )
#     return detections, depth_map


# # ── Health ────────────────────────────────────────────────────────────────────
# @app.get("/")
# def root():
#     return {"message": "UVMS v4.0 running", "gpu": torch.cuda.is_available()}


# @app.get("/health")
# def health(request: Request):
#     bd = request.app.state.building_detector
#     return {
#         "status":                "online",
#         "version":               "4.0.0",
#         "gpu":                   torch.cuda.is_available(),
#         "device":                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
#         "building_model_loaded": getattr(bd, "_loaded", True),
#         "endpoints": [
#             "/detect", "/buildings", "/waste",
#             "/weapon", "/furniture", "/depth", "/analyze", "/urban"
#         ]
#     }


# # ── Vehicle detection ─────────────────────────────────────────────────────────
# @app.post("/detect")
# async def detect(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.40, ge=0.1, le=0.95),
# ):
#     detector  = request.app.state.detector
#     depth_est = request.app.state.depth_est
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="vehicle")
#     detections, _ = add_ground_clearance(detections, image, depth_est)
#     annotated  = detector.draw(image, detections, mode="vehicle")
#     return JSONResponse({
#         "count":           len(detections),
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── Building detection ────────────────────────────────────────────────────────
# @app.post("/buildings")
# async def buildings(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.40, ge=0.1, le=0.95),
# ):
#     building_detector = request.app.state.building_detector
#     image      = require_image(await file.read())
#     detections = building_detector.detect(image, confidence)
#     annotated  = building_detector.draw(image, detections)
#     by_type    = {}
#     for d in detections:
#         by_type[d["label"]] = by_type.get(d["label"], 0) + 1
#     return JSONResponse({
#         "count":           len(detections),
#         "by_type":         by_type,
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── Waste detection ───────────────────────────────────────────────────────────
# @app.post("/waste")
# async def waste(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.25, ge=0.1, le=0.95),
# ):
#     detector   = request.app.state.detector
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="waste")
#     annotated  = detector.draw(image, detections, mode="waste")
#     summary    = MultiDetector.waste_summary(detections)
#     return JSONResponse({
#         "count":                   summary["total"],
#         "recyclable_count":        summary["recyclable_count"],
#         "non_recyclable_count":    summary["non_recyclable_count"],
#         "recyclability_rate":      summary["recyclability_rate"],
#         "avg_recyclability_score": summary["avg_recyclability_score"],
#         "by_item":                 summary["by_item"],
#         "by_material":             summary["by_material"],
#         "special_notes":           summary["special_notes"],
#         "detections":              detections,
#         "annotated_image":         b64(annotated),
#         "original_image":          b64(image),
#     })


# # ── Weapon detection ──────────────────────────────────────────────────────────
# @app.post("/weapon")
# async def weapon(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.35, ge=0.1, le=0.95),
# ):
#     detector   = request.app.state.detector
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="weapon")
#     annotated  = detector.draw(image, detections, mode="weapon")
#     alert      = len(detections) > 0
#     high_threat = any(d.get("threat_level") == "🔴 HIGH" for d in detections)
#     return JSONResponse({
#         "alert":           alert,
#         "high_threat":     high_threat,
#         "count":           len(detections),
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── Furniture (direct endpoint — mirrors router but no trailing slash) ─────────
# @app.post("/furniture")
# async def furniture(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.30, ge=0.1, le=0.95),
# ):
#     detector   = request.app.state.detector
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="furniture")
#     annotated  = detector.draw(image, detections, mode="furniture")
#     counts     = {}
#     for d in detections:
#         counts[d["label"]] = counts.get(d["label"], 0) + 1
#     return JSONResponse({
#         "count":           len(detections),
#         "by_type":         counts,
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── Depth ─────────────────────────────────────────────────────────────────────
# @app.post("/depth")
# async def depth(request: Request, file: UploadFile = File(...)):
#     depth_est = request.app.state.depth_est
#     image     = require_image(await file.read())
#     depth_map = depth_est.estimate(image)
#     colored   = depth_est.colorize(depth_map)
#     return JSONResponse({
#         "depth_image":    b64(colored),
#         "original_image": b64(image),
#     })


# # ── Analyze ───────────────────────────────────────────────────────────────────
# @app.post("/analyze")
# async def analyze(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.40),
#     mode:       str        = Query("vehicle"),
# ):
#     detector  = request.app.state.detector
#     depth_est = request.app.state.depth_est
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode=mode)
#     depth_map  = depth_est.estimate(image)
#     colored    = depth_est.colorize(depth_map)
#     if mode == "vehicle":
#         for d in detections:
#             x1, y1, x2, y2 = d["bbox"]
#             cx = (x1+x2)//2
#             car_y    = min(y2,      depth_map.shape[0]-1)
#             ground_y = min(y2+15,   depth_map.shape[0]-1)
#             d["ground_clearance_cm"] = round(
#                 abs(int(depth_map[ground_y, cx]) - int(depth_map[car_y, cx])) * 1.5, 1
#             )
#     annotated = detector.draw(image, detections, mode=mode)
#     payload = {
#         "mode":            mode,
#         "count":           len(detections),
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "depth_image":     b64(colored),
#     }
#     if mode == "waste":
#         s = MultiDetector.waste_summary(detections)
#         payload.update({k: s[k] for k in
#                         ["recyclable_count","non_recyclable_count",
#                          "recyclability_rate","by_item","by_material","special_notes"]})
#     if mode == "weapon":
#         payload["alert"]       = len(detections) > 0
#         payload["high_threat"] = any(d.get("threat_level") == "🔴 HIGH" for d in detections)
#     return JSONResponse(payload)


# # ── Urban ─────────────────────────────────────────────────────────────────────
# @app.post("/urban")
# async def urban(
#     request:         Request,
#     file:            UploadFile = File(...),
#     car_confidence:  float      = Query(0.40),
#     bldg_confidence: float      = Query(0.40),
#     with_depth:      bool       = Query(True),
# ):
#     detector          = request.app.state.detector
#     building_detector = request.app.state.building_detector
#     depth_est         = request.app.state.depth_est
#     image     = require_image(await file.read())
#     cars      = detector.detect(image, car_confidence, mode="vehicle")
#     buildings = building_detector.detect(image, bldg_confidence)
#     depth_map = None
#     if cars:
#         cars, depth_map = add_ground_clearance(cars, image, depth_est)
#     annotated = building_detector.draw(image, buildings)
#     annotated = detector.draw(annotated, cars, mode="vehicle")
#     car_types  = {}
#     bldg_types = {}
#     for d in cars:      car_types[d["label"]]  = car_types.get(d["label"],  0) + 1
#     for d in buildings: bldg_types[d["label"]] = bldg_types.get(d["label"], 0) + 1
#     payload = {
#         "cars":          {"count": len(cars),      "by_type": car_types,  "detections": cars},
#         "buildings":     {"count": len(buildings), "by_type": bldg_types, "detections": buildings},
#         "total_objects": len(cars) + len(buildings),
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     }
#     if with_depth and depth_map is not None:
#         payload["depth_image"] = b64(depth_est.colorize(depth_map))
#     return JSONResponse(payload)


# # ── Static pages ──────────────────────────────────────────────────────────────
# @app.get("/web")
# def web():
#     return FileResponse("templates/index.html")


# @app.get("/app")
# def app_page():
#     return FileResponse("templates/app.html")


# # needed for health endpoint
# from fastapi import Request






# """
# main.py — UVMS v5.0
# Added: Road Damage Detection + Wildlife Detection
# """

# import traceback
# import cv2
# import numpy as np
# import base64
# import torch

# from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, FileResponse
# from contextlib import asynccontextmanager

# from models.detector          import MultiDetector
# from models.building_detector import BuildingDetector
# from models.depth             import DepthEstimator
# from models.road_damage       import RoadDamageDetector
# from models.wildlife          import WildlifeDetector


# # ── Lifespan ──────────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("🚀 Loading AI models...")
#     try:
#         app.state.detector          = MultiDetector(model_size="l")
#         print("✅ detector loaded")
#         app.state.building_detector = BuildingDetector()
#         print("✅ building detector loaded")
#         app.state.depth_est         = DepthEstimator()
#         print("✅ depth estimator loaded")
#         app.state.road_detector     = RoadDamageDetector()
#         print("✅ road damage detector loaded")
#         # Wildlife reuses YOLO model — no extra download!
#         app.state.wildlife_detector = WildlifeDetector(
#             app.state.detector.model
#         )
#         print("✅ wildlife detector loaded")
#     except Exception:
#         print("❌ Model load failed:")
#         traceback.print_exc()
#         raise
#     yield


# # ── App ───────────────────────────────────────────────────────
# app = FastAPI(
#     title="UVMS API v5.0",
#     version="5.0.0",
#     lifespan=lifespan,
#     description="""
# ## Urban Vision Management System v5.0

# | Endpoint       | Purpose                        |
# |----------------|--------------------------------|
# | POST /detect   | 🚗 Vehicle detection           |
# | POST /furniture| 🛋️ Furniture detection         |
# | POST /waste    | ♻️ Waste detection             |
# | POST /buildings| 🏢 Building detection          |
# | POST /wildlife | 🦁 Animal & bird detection     |
# | POST /road     | 🚧 Road damage + measurements  |
# | POST /depth    | 🌊 Depth estimation            |
# | POST /analyze  | ⚡ Any mode + depth            |
# """,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# from routes.furniture import router as furniture_router
# from routes.building  import router as building_router
# from routes.waste     import router as waste_router

# app.include_router(furniture_router)
# app.include_router(building_router)
# app.include_router(waste_router)


# # ── Helpers ───────────────────────────────────────────────────
# def b64(img):
#     _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     return base64.b64encode(buf).decode()


# def read_img(file_bytes: bytes):
#     arr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if img is None:
#         try:
#             from PIL import Image
#             import io
#             pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#             img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
#         except Exception:
#             return None
#     return img


# def require_image(file_bytes: bytes):
#     img = read_img(file_bytes)
#     if img is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot read image. Supported: JPG, PNG, WEBP, BMP"
#         )
#     return img


# def add_ground_clearance(detections, image, depth_est):
#     if not detections:
#         return detections, None
#     depth_map = depth_est.estimate(image)
#     for d in detections:
#         x1, y1, x2, y2 = d["bbox"]
#         cx       = (x1 + x2) // 2
#         car_y    = min(y2,      depth_map.shape[0] - 1)
#         ground_y = min(y2 + 15, depth_map.shape[0] - 1)
#         d["ground_clearance_cm"] = round(
#             abs(int(depth_map[ground_y, cx]) -
#                 int(depth_map[car_y,    cx])) * 1.5, 1
#         )
#     return detections, depth_map


# # ── Health ────────────────────────────────────────────────────
# @app.get("/")
# def root():
#     return {"message": "UVMS v5.0 running", "gpu": torch.cuda.is_available()}


# @app.get("/health")
# def health(request: Request):
#     bd = request.app.state.building_detector
#     rd = request.app.state.road_detector
#     wd = request.app.state.wildlife_detector
#     return {
#         "status":               "online",
#         "version":              "5.0.0",
#         "gpu":                  torch.cuda.is_available(),
#         "device":               torch.cuda.get_device_name(0)
#                                 if torch.cuda.is_available() else "CPU",
#         "building_model_loaded": getattr(bd, "_loaded", True),
#         "road_model_loaded":     getattr(rd, "_loaded", False),
#         "wildlife_model_loaded": getattr(wd, "_loaded", False),
#         "endpoints": [
#             "/detect", "/furniture", "/waste",
#             "/buildings", "/wildlife", "/road",
#             "/depth", "/analyze"
#         ]
#     }


# # ── Vehicle detection ─────────────────────────────────────────
# @app.post("/detect")
# async def detect(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.40, ge=0.1, le=0.95),
# ):
#     detector  = request.app.state.detector
#     depth_est = request.app.state.depth_est
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="vehicle")
#     detections, _ = add_ground_clearance(detections, image, depth_est)
#     annotated  = detector.draw(image, detections, mode="vehicle")
#     return JSONResponse({
#         "count":           len(detections),
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── Building detection ────────────────────────────────────────
# @app.post("/buildings")
# async def buildings(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.40, ge=0.1, le=0.95),
# ):
#     building_detector = request.app.state.building_detector
#     image      = require_image(await file.read())
#     detections = building_detector.detect(image, confidence)
#     annotated  = building_detector.draw(image, detections)
#     by_type    = {}
#     for d in detections:
#         by_type[d["label"]] = by_type.get(d["label"], 0) + 1
#     return JSONResponse({
#         "count":           len(detections),
#         "by_type":         by_type,
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── Waste detection ───────────────────────────────────────────
# @app.post("/waste")
# async def waste(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.25, ge=0.1, le=0.95),
# ):
#     detector   = request.app.state.detector
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="waste")
#     annotated  = detector.draw(image, detections, mode="waste")
#     summary    = MultiDetector.waste_summary(detections)
#     return JSONResponse({
#         "count":                   summary["total"],
#         "recyclable_count":        summary["recyclable_count"],
#         "non_recyclable_count":    summary["non_recyclable_count"],
#         "recyclability_rate":      summary["recyclability_rate"],
#         "avg_recyclability_score": summary["avg_recyclability_score"],
#         "by_item":                 summary["by_item"],
#         "by_material":             summary["by_material"],
#         "special_notes":           summary["special_notes"],
#         "detections":              detections,
#         "annotated_image":         b64(annotated),
#         "original_image":          b64(image),
#     })


# # ── Furniture detection ───────────────────────────────────────
# @app.post("/furniture")
# async def furniture(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.30, ge=0.1, le=0.95),
# ):
#     detector   = request.app.state.detector
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode="furniture")
#     annotated  = detector.draw(image, detections, mode="furniture")
#     counts     = {}
#     for d in detections:
#         counts[d["label"]] = counts.get(d["label"], 0) + 1
#     return JSONResponse({
#         "count":           len(detections),
#         "by_type":         counts,
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── 🦁 Wildlife detection ─────────────────────────────────────
# @app.post("/wildlife")
# async def wildlife(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.30, ge=0.1, le=0.95),
# ):
#     wildlife_detector = request.app.state.wildlife_detector
#     image      = require_image(await file.read())
#     detections = wildlife_detector.detect(image, confidence)
#     annotated  = wildlife_detector.draw(image, detections)
#     summary    = WildlifeDetector.wildlife_summary(detections)
#     return JSONResponse({
#         "count":           summary["total"],
#         "by_animal":       summary["by_animal"],
#         "by_group":        summary["by_group"],
#         "groups_found":    summary["groups_found"],
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "original_image":  b64(image),
#     })


# # ── 🚧 Road damage detection ──────────────────────────────────
# @app.post("/road")
# async def road_damage(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.30, ge=0.1, le=0.95),
#     with_depth: bool       = Query(True),
# ):
#     road_detector = request.app.state.road_detector
#     depth_est     = request.app.state.depth_est

#     if not road_detector._loaded:
#         raise HTTPException(
#             503, "Road damage model not loaded — check startup logs"
#         )

#     image     = require_image(await file.read())
#     depth_map = depth_est.estimate(image) if with_depth else None
#     detections = road_detector.detect(image, confidence, depth_map)
#     annotated  = road_detector.draw(image, detections)
#     summary    = RoadDamageDetector.damage_summary(detections)
#     annotated  = road_detector.draw(image, detections)

#     payload = {
#         "count":                  summary["total"],
#         "road_condition":         summary["road_condition"],
#         "urgent_count":           summary["urgent_count"],
#         "monitor_count":          summary["monitor_count"],
#         "total_damage_area_cm2":  summary["total_damage_area_cm2"],
#         "by_type":                summary["by_type"],
#         "detections":             detections,
#         "annotated_image":        b64(annotated),
#         "original_image":         b64(image),
#     }
#     if with_depth and depth_map is not None:
#         payload["depth_image"] = b64(depth_est.colorize(depth_map))

#     return JSONResponse(payload)


# # ── Depth ─────────────────────────────────────────────────────
# @app.post("/depth")
# async def depth(request: Request, file: UploadFile = File(...)):
#     depth_est = request.app.state.depth_est
#     image     = require_image(await file.read())
#     depth_map = depth_est.estimate(image)
#     colored   = depth_est.colorize(depth_map)
#     return JSONResponse({
#         "depth_image":    b64(colored),
#         "original_image": b64(image),
#     })


# # ── Analyze ───────────────────────────────────────────────────
# @app.post("/analyze")
# async def analyze(
#     request:    Request,
#     file:       UploadFile = File(...),
#     confidence: float      = Query(0.40),
#     mode:       str        = Query("vehicle"),
# ):
#     detector  = request.app.state.detector
#     depth_est = request.app.state.depth_est
#     image      = require_image(await file.read())
#     detections = detector.detect(image, confidence, mode=mode)
#     depth_map  = depth_est.estimate(image)
#     colored    = depth_est.colorize(depth_map)
#     if mode == "vehicle":
#         for d in detections:
#             x1, y1, x2, y2 = d["bbox"]
#             cx       = (x1+x2)//2
#             car_y    = min(y2,    depth_map.shape[0]-1)
#             ground_y = min(y2+15, depth_map.shape[0]-1)
#             d["ground_clearance_cm"] = round(
#                 abs(int(depth_map[ground_y, cx]) -
#                     int(depth_map[car_y,    cx])) * 1.5, 1
#             )
#     annotated = detector.draw(image, detections, mode=mode)
#     payload = {
#         "mode":            mode,
#         "count":           len(detections),
#         "detections":      detections,
#         "annotated_image": b64(annotated),
#         "depth_image":     b64(colored),
#     }
#     return JSONResponse(payload)


# # ── Static pages ──────────────────────────────────────────────
# @app.get("/web")
# def web():
#     return FileResponse("templates/index.html")


# @app.get("/app")
# def app_page():
#     return FileResponse("templates/app1.html")  








# new one 
"""
main.py — UVMS v5.0   (FIXED)
Bugs fixed:
  1. Duplicate road annotated= line removed
  2. Router conflict removed (routes.* includes conflicted with direct endpoints)
  3. road_detector._loaded replaced with is_ready()
  4. with_depth=False depth_map=None guard added to /road
  5. waste_summary / damage_summary / wildlife_summary must be @staticmethod
     (added note — ensure those methods have the decorator in their files)
"""

import traceback
import cv2
import numpy as np
import base64
import torch

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager

from models.detector          import MultiDetector
from models.building_detector import BuildingDetector
from models.depth             import DepthEstimator
from models.road_damage       import RoadDamageDetector
from models.wildlife          import WildlifeDetector


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading AI models...")
    try:
        app.state.detector          = MultiDetector(model_size="l")
        print("detector loaded")
        app.state.building_detector = BuildingDetector()
        print("building detector loaded")
        app.state.depth_est         = DepthEstimator()
        print("depth estimator loaded")
        app.state.road_detector     = RoadDamageDetector()
        print("road damage detector loaded")
        app.state.wildlife_detector = WildlifeDetector(
            app.state.detector.model
        )
        print("wildlife detector loaded")
    except Exception:
        print("Model load failed:")
        traceback.print_exc()
        raise
    yield


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="UVMS API v5.0",
    version="5.0.0",
    lifespan=lifespan,
    description="""
## Urban Vision Management System v5.0

| Endpoint        | Purpose                        |
|-----------------|--------------------------------|
| POST /detect    | Vehicle detection              |
| POST /furniture | Furniture detection            |
| POST /waste     | Waste detection                |
| POST /buildings | Building detection             |
| POST /wildlife  | Animal & bird detection        |
| POST /road      | Road damage + measurements     |
| POST /depth     | Depth estimation               |
| POST /analyze   | Any mode + depth               |
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────
# BUG 2 FIXED: REMOVED the router includes below.
#
# Your original code had:
#   from routes.furniture import router as furniture_router   <- included
#   app.include_router(furniture_router)                      <- included
# AND ALSO defined @app.post("/furniture") directly below.
# FastAPI sees two routes for the same path -> the last one silently wins
# but it can cause 422 / unexpected behaviour depending on FastAPI version.
#
# RULE: pick ONE approach. Since all endpoints are written directly here,
# the router includes are removed. If you want to split into route files,
# remove the direct endpoint definitions instead.
# ────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────
def b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


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


def require_image(file_bytes: bytes) -> np.ndarray:
    img = read_img(file_bytes)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot read image. Supported: JPG, PNG, WEBP, BMP, AVIF"
        )
    return img


def add_ground_clearance(detections, image, depth_est):
    """Adds ground_clearance_cm to each vehicle detection."""
    if not detections:
        return detections, None
    depth_map = depth_est.estimate(image)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cx       = (x1 + x2) // 2
        car_y    = min(y2,      depth_map.shape[0] - 1)
        ground_y = min(y2 + 15, depth_map.shape[0] - 1)
        d["ground_clearance_cm"] = round(
            abs(int(depth_map[ground_y, cx]) -
                int(depth_map[car_y,    cx])) * 1.5, 1
        )
    return detections, depth_map


# ── Health ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "UVMS v5.0 running", "gpu": torch.cuda.is_available()}


@app.get("/health")
def health(request: Request):
    rd = request.app.state.road_detector
    wd = request.app.state.wildlife_detector
    bd = request.app.state.building_detector
    return {
        "status":                "online",
        "version":               "5.0.0",
        "gpu":                   torch.cuda.is_available(),
        "device":                torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "CPU",
        # BUG 3 FIXED: use is_ready() instead of ._loaded directly
        "road_model_loaded":     rd.is_ready() if hasattr(rd, "is_ready")
                                 else getattr(rd, "_loaded", False),
        "wildlife_model_loaded": getattr(wd, "_loaded", False),
        "building_model_loaded": getattr(bd, "_loaded", True),
        "endpoints": [
            "/detect", "/furniture", "/waste",
            "/buildings", "/wildlife", "/road",
            "/depth", "/analyze"
        ],
    }


# ── Vehicle detection ─────────────────────────────────────────
@app.post("/detect")
async def detect(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.40, ge=0.1, le=0.95),
):
    detector  = request.app.state.detector
    depth_est = request.app.state.depth_est
    image      = require_image(await file.read())
    detections = detector.detect(image, confidence, mode="vehicle")
    detections, _ = add_ground_clearance(detections, image, depth_est)
    annotated  = detector.draw(image, detections, mode="vehicle")
    return JSONResponse({
        "count":           len(detections),
        "detections":      detections,
        "annotated_image": b64(annotated),
        "original_image":  b64(image),
    })


# ── Building detection ────────────────────────────────────────
@app.post("/buildings")
async def buildings(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.40, ge=0.1, le=0.95),
):
    building_detector = request.app.state.building_detector
    image      = require_image(await file.read())
    detections = building_detector.detect(image, confidence)
    annotated  = building_detector.draw(image, detections)
    by_type    = {}
    for d in detections:
        by_type[d["label"]] = by_type.get(d["label"], 0) + 1
    return JSONResponse({
        "count":           len(detections),
        "by_type":         by_type,
        "detections":      detections,
        "annotated_image": b64(annotated),
        "original_image":  b64(image),
    })


# ── Waste detection ───────────────────────────────────────────
@app.post("/waste")
async def waste(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.25, ge=0.1, le=0.95),
):
    detector   = request.app.state.detector
    image      = require_image(await file.read())
    detections = detector.detect(image, confidence, mode="waste")
    annotated  = detector.draw(image, detections, mode="waste")
    # BUG 5 NOTE: ensure MultiDetector.waste_summary is decorated @staticmethod
    summary    = MultiDetector.waste_summary(detections)
    return JSONResponse({
        "count":                   summary["total"],
        "recyclable_count":        summary["recyclable_count"],
        "non_recyclable_count":    summary["non_recyclable_count"],
        "recyclability_rate":      summary["recyclability_rate"],
        "avg_recyclability_score": summary["avg_recyclability_score"],
        "by_item":                 summary["by_item"],
        "by_material":             summary["by_material"],
        "special_notes":           summary["special_notes"],
        "detections":              detections,
        "annotated_image":         b64(annotated),
        "original_image":          b64(image),
    })


# ── Furniture detection ───────────────────────────────────────
@app.post("/furniture")
async def furniture(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.30, ge=0.1, le=0.95),
):
    detector   = request.app.state.detector
    image      = require_image(await file.read())
    detections = detector.detect(image, confidence, mode="furniture")
    annotated  = detector.draw(image, detections, mode="furniture")
    counts     = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return JSONResponse({
        "count":           len(detections),
        "by_type":         counts,
        "detections":      detections,
        "annotated_image": b64(annotated),
        "original_image":  b64(image),
    })


# ── Wildlife detection ────────────────────────────────────────
@app.post("/wildlife")
async def wildlife(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.30, ge=0.1, le=0.95),
):
    wildlife_detector = request.app.state.wildlife_detector
    image      = require_image(await file.read())
    detections = wildlife_detector.detect(image, confidence)
    annotated  = wildlife_detector.draw(image, detections)
    # BUG 7 NOTE: ensure WildlifeDetector.wildlife_summary is @staticmethod
    summary    = WildlifeDetector.wildlife_summary(detections)
    return JSONResponse({
        "count":           summary["total"],
        "by_animal":       summary["by_animal"],
        "by_group":        summary["by_group"],
        "groups_found":    summary["groups_found"],
        "detections":      detections,
        "annotated_image": b64(annotated),
        "original_image":  b64(image),
    })


# ── Road damage detection ─────────────────────────────────────
@app.post("/road")
async def road_damage(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.30, ge=0.1, le=0.95),
    with_depth: bool       = Query(True),
):
    road_detector = request.app.state.road_detector
    depth_est     = request.app.state.depth_est

    # BUG 3 FIXED: use is_ready() not ._loaded
    ready = road_detector.is_ready() if hasattr(road_detector, "is_ready") \
            else getattr(road_detector, "_loaded", False)
    if not ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Road damage model not loaded. "
                "Place pothole_yolov8.pt in app/models/ and restart server."
            )
        )

    image = require_image(await file.read())

    # BUG 4 FIXED: only run depth when with_depth=True; pass None safely
    depth_map = depth_est.estimate(image) if with_depth else None

    detections = road_detector.detect(image, confidence, depth_map)

    # BUG 6 NOTE: ensure RoadDamageDetector.damage_summary is @staticmethod
    summary   = RoadDamageDetector.damage_summary(detections)

    # BUG 1 FIXED: draw called only ONCE (original had it twice)
    annotated = road_detector.draw(image, detections)

    payload = {
        "count":                 summary["total"],
        "road_condition":        summary["road_condition"],
        "urgent_count":          summary["urgent_count"],
        "monitor_count":         summary["monitor_count"],
        "total_damage_area_cm2": summary["total_damage_area_cm2"],
        "by_type":               summary["by_type"],
        "detections":            detections,
        "annotated_image":       b64(annotated),
        "original_image":        b64(image),
    }
    if with_depth and depth_map is not None:
        payload["depth_image"] = b64(depth_est.colorize(depth_map))

    return JSONResponse(payload)


# ── Depth ─────────────────────────────────────────────────────
@app.post("/depth")
async def depth(request: Request, file: UploadFile = File(...)):
    depth_est = request.app.state.depth_est
    image     = require_image(await file.read())
    depth_map = depth_est.estimate(image)
    colored   = depth_est.colorize(depth_map)
    return JSONResponse({
        "depth_image":    b64(colored),
        "original_image": b64(image),
    })


# ── Analyze ───────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    request:    Request,
    file:       UploadFile = File(...),
    confidence: float      = Query(0.40),
    mode:       str        = Query(
        "vehicle",
        description="vehicle | furniture | waste | weapon"
    ),
):
    detector  = request.app.state.detector
    depth_est = request.app.state.depth_est
    image      = require_image(await file.read())
    detections = detector.detect(image, confidence, mode=mode)
    depth_map  = depth_est.estimate(image)
    colored    = depth_est.colorize(depth_map)

    if mode == "vehicle":
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx       = (x1 + x2) // 2
            car_y    = min(y2,      depth_map.shape[0] - 1)
            ground_y = min(y2 + 15, depth_map.shape[0] - 1)
            d["ground_clearance_cm"] = round(
                abs(int(depth_map[ground_y, cx]) -
                    int(depth_map[car_y,    cx])) * 1.5, 1
            )

    annotated = detector.draw(image, detections, mode=mode)
    return JSONResponse({
        "mode":            mode,
        "count":           len(detections),
        "detections":      detections,
        "annotated_image": b64(annotated),
        "depth_image":     b64(colored),
    })


# ── Static pages ──────────────────────────────────────────────
@app.get("/web")
def web():
    return FileResponse("templates/index.html")


@app.get("/app")
def app_page():
    return FileResponse("templates/app1.html")