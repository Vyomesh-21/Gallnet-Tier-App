"""FastAPI Backend for GallNet Triage.

Provides research endpoints:
  - GET /health: Model and system health
  - GET /model-info: Architecture, labels, tiers, and verification status
  - POST /predict: Real model inference and three-tier triage analysis
"""

import io
import os
import logging
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from model_service import ModelService, MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gallnet.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = STATIC_DIR / "assets"

app = FastAPI(
    title="GallNet Triage API",
    description="Automated Surgical Triage for Gallbladder Perforation Risk (Research Prototype)",
    version="1.0.0"
)

if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Allow local frontend origins for development plus production origins from environment
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

frontend_origin_env = os.getenv("FRONTEND_ORIGIN") or os.getenv("ALLOWED_ORIGINS", "")
if frontend_origin_env:
    for raw_origin in frontend_origin_env.split(","):
        cleaned = raw_origin.strip().rstrip("/")
        if cleaned and cleaned not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(cleaned)
            logger.info("Added allowed CORS origin: %s", cleaned)

cors_kwargs: Dict[str, Any] = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["*"],
}

origin_regex = os.getenv(
    "FRONTEND_ORIGIN_REGEX",
    r"https://.*\.onrender\.com|https://.*\.vercel\.app"
)
if origin_regex:
    cors_kwargs["allow_origin_regex"] = origin_regex

app.add_middleware(
    CORSMiddleware,
    **cors_kwargs,
)

# Initialize model service once at startup
service = ModelService()

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@app.get("/", include_in_schema=False)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"message": "GallNet Triage API is online. Visit /docs for interactive documentation."}



@app.get("/health", summary="System Health and Model Readiness")
def health() -> Dict[str, Any]:
    return {
        "status": "online",
        "model_loaded": service.ready,
        "model_name": MODEL_NAME,
        "model_error": service.error,
        "mapping_verified": service.mapping_verified,
        "mapping_status": "VERIFIED" if service.mapping_verified else "UNVERIFIED"
    }


@app.get("/model-info", summary="Model Metadata and Triage Configuration")
def model_info() -> Dict[str, Any]:
    from tier_mapping import TIER_MEMBERS
    return {
        "model_name": MODEL_NAME,
        "model_loaded": service.ready,
        "input_shape": [None, 128, 128, 3],
        "output_classes": 9,
        "candidate_labels": service.labels,
        "mapping_verified": service.mapping_verified,
        "mapping_status": "VERIFIED" if service.mapping_verified else "UNVERIFIED",
        "mapping_details": service.mapping_data,
        "tiers": {
            k: {
                "name": v["name"],
                "description": v["description"],
                "classes": v["classes"]
            }
            for k, v in TIER_MEMBERS.items()
        },
        "preprocessing": "OpenCV BGR, resized to (128, 128), normalized to [0, 1] by / 255.0",
        "explainability": {
            "method": "Grad-CAM",
            "layer": service.last_conv_layer_name,
            "status": "active" if service.grad_model is not None else "unavailable"
        },
        "error": service.error
    }


@app.post("/predict", summary="Run Real Inference and Triage Analysis")
def predict(file: UploadFile = File(...)):
    if not service.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Real pretrained inference is unavailable: {service.error}"
        )

    # Validate content type
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Please upload a PNG or JPEG ultrasound image."
        )

    # Read bytes synchronously without persisting to disk (preserving patient privacy)
    try:
        if hasattr(file, "file") and hasattr(file.file, "read"):
            raw = file.file.read(MAX_IMAGE_BYTES + 1)
        else:
            raw = file.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {str(exc)}"
        )

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed size of 10 MB."
        )

    # Validate image integrity and dimensions
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            if width < 32 or height < 32:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image dimensions ({width}x{height}) are too small. Minimum resolution is 32x32 pixels."
                )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupt or unreadable image file: {str(exc)}"
        )

    # Execute REAL model inference
    try:
        result = service.predict(raw)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as exc:
        logger.exception("Prediction failed during execution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting GallNet API server on 0.0.0.0:%d", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

