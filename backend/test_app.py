"""Comprehensive Test Suite for GallNet Triage.

Covers:
  - Model loading and architecture
  - OpenCV / Pillow BGR preprocessing
  - Model output shapes, finite probabilities, sum == 1.0
  - Three-tier aggregation and Review Required handling
  - Grad-CAM heatmap generation
  - FastAPI endpoints (/health, /model-info, /predict)
  - Error handling for invalid/corrupted uploads
  - Real inference on a valid sample ultrasound image
"""

import io
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi import UploadFile, HTTPException

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from tier_mapping import aggregate_triage, CANDIDATE_ORDER, TIER_MEMBERS
from model_service import ModelService
from main import health, model_info, predict, serve_index


def test_api_serve_index():
    res = serve_index()
    assert res is not None



def create_sample_ultrasound_image(width=256, height=256, format="JPEG") -> bytes:
    """Generates a valid synthetic ultrasound-like image with acoustic speckle and organ contour."""
    # Synthetic grayscale acoustic speckle texture (Rayleigh-like distribution)
    rng = np.random.RandomState(42)
    noise1 = rng.normal(80, 25, (height, width))
    noise2 = rng.normal(60, 20, (height, width))
    speckle = np.sqrt(noise1**2 + noise2**2)
    
    # Simulate gallbladder lumen (anechoic / dark region in center)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    lumen_mask = ((x - center_x) ** 2) / (60**2) + ((y - center_y) ** 2) / (40**2) <= 1.0
    speckle[lumen_mask] *= 0.15  # Dark acoustic fluid

    # Hyperechoic gallbladder wall (bright boundary)
    wall_mask = (
        (((x - center_x) ** 2) / (66**2) + ((y - center_y) ** 2) / (46**2) <= 1.0) &
        ~lumen_mask
    )
    speckle[wall_mask] = np.clip(speckle[wall_mask] * 2.2 + 80, 0, 255)

    img_array = np.clip(speckle, 0, 255).astype(np.uint8)
    img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
    
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format=format)
    return buf.getvalue()


# -------------------------------------------------------------
# 1. Tier Aggregation Tests (Verified Modulo Mapping Order)
# -------------------------------------------------------------
def test_tier_0_aggregation():
    # Verified Intact classes: 0 (Wall thickening), 1 (Gallstones), 3 (Cholecystitis), 6 (Polyps), 7 (Adenomyomatosis)
    probs = np.zeros(9, dtype=np.float64)
    probs[0] = 0.20  # Wall thickening
    probs[1] = 0.20  # Gallstones
    probs[3] = 0.15  # Cholecystitis
    probs[6] = 0.15  # Polyps
    probs[7] = 0.10  # Adenomyomatosis
    probs[4] = 0.05  # Membranous (Tier 1)
    probs[5] = 0.05  # Perforation (Tier 2)
    probs[2] = 0.05  # Abdomen (Review)
    probs[8] = 0.05  # Carcinoma (Review)
    res = aggregate_triage(probs, CANDIDATE_ORDER, mapping_verified=True)
    assert res["assigned_triage_category"] == "Tier_0"
    assert res["tier_scores"]["Tier_0"] == pytest.approx(0.80)
    assert res["tier_scores"]["Tier_1"] == pytest.approx(0.05)
    assert res["tier_scores"]["Tier_2"] == pytest.approx(0.05)
    assert res["tier_scores"]["Review_Required"] == pytest.approx(0.10)
    total_tiers = sum(res["tier_scores"].values())
    assert total_tiers == pytest.approx(1.0)


def test_tier_1_gangrenous_cholecystitis():
    # Verified Index 4 is Membranous and gangrenous cholecystitis
    probs = np.zeros(9, dtype=np.float64)
    probs[4] = 0.70
    probs[0] = 0.05
    probs[1] = 0.05
    probs[2] = 0.05
    probs[3] = 0.05
    probs[5] = 0.02
    probs[6] = 0.03
    probs[7] = 0.02
    probs[8] = 0.03
    res = aggregate_triage(probs, CANDIDATE_ORDER, mapping_verified=True)
    assert res["assigned_triage_category"] == "Tier_1"
    assert res["dominant_original_prediction"]["class_name"] == "Membranous and gangrenous cholecystitis"
    assert res["tier_scores"]["Tier_1"] == pytest.approx(0.70)


def test_tier_2_perforation():
    # Verified Index 5 is Perforation
    probs = np.zeros(9, dtype=np.float64)
    probs[5] = 0.85
    probs[0] = 0.02
    probs[1] = 0.02
    probs[2] = 0.02
    probs[3] = 0.02
    probs[4] = 0.02
    probs[6] = 0.01
    probs[7] = 0.02
    probs[8] = 0.02
    res = aggregate_triage(probs, CANDIDATE_ORDER, mapping_verified=True)
    assert res["assigned_triage_category"] == "Tier_2"
    assert res["dominant_original_prediction"]["class_name"] == "Perforation"
    assert res["tier_scores"]["Tier_2"] == pytest.approx(0.85)


def test_review_required_for_carcinoma():
    # Verified Index 8 is Carcinoma
    probs = np.zeros(9, dtype=np.float64)
    probs[8] = 0.70
    probs[2] = 0.05  # Abdomen (Review)
    probs[0] = 0.05
    probs[1] = 0.05
    probs[3] = 0.05
    probs[4] = 0.02
    probs[5] = 0.03
    probs[6] = 0.02
    probs[7] = 0.03
    res = aggregate_triage(probs, CANDIDATE_ORDER, mapping_verified=True)
    assert res["assigned_triage_category"] == "Review_Required"
    assert res["dominant_original_prediction"]["class_name"] == "Carcinoma"
    assert res["tier_scores"]["Review_Required"] == pytest.approx(0.75)


def test_review_required_for_abdomen():
    # Verified Index 2 is Abdomen and retroperitoneum
    probs = np.zeros(9, dtype=np.float64)
    probs[2] = 0.65
    probs[0] = 0.05
    probs[1] = 0.05
    probs[3] = 0.05
    probs[4] = 0.05
    probs[5] = 0.05
    probs[6] = 0.03
    probs[7] = 0.03
    probs[8] = 0.04
    res = aggregate_triage(probs, CANDIDATE_ORDER, mapping_verified=True)
    assert res["assigned_triage_category"] == "Review_Required"
    assert res["dominant_original_prediction"]["class_name"] == "Abdomen and retroperitoneum"


def test_unverified_mapping_blocks_triage():
    probs = np.array([0.1] * 8 + [0.2])
    res = aggregate_triage(probs, CANDIDATE_ORDER, mapping_verified=False)
    assert res["mapping_status"] == "UNVERIFIED"
    assert res["triage_blocked"] is True
    assert res["verification_warning"] is not None


# -------------------------------------------------------------
# 2. Pretrained Model Loading and Real Inference Tests
# -------------------------------------------------------------
def test_pretrained_model_loads_successfully():
    service = ModelService()
    assert service.ready is True, f"Model failed to load: {service.error}"
    assert service.error is None
    assert service.model is not None


def test_model_shapes():
    service = ModelService()
    in_shape = service.model.input_shape
    out_shape = service.model.output_shape
    assert in_shape == (None, 128, 128, 3)
    out_dim = out_shape[-1] if not isinstance(out_shape, list) else out_shape[0][-1]
    assert out_dim == 9


def test_bgr_preprocessing():
    service = ModelService()
    sample_bytes = create_sample_ultrasound_image(200, 150)
    batch_bgr, rgb_128, orig_w, orig_h = service.preprocess_image(sample_bytes)
    
    assert batch_bgr.shape == (1, 128, 128, 3)
    assert batch_bgr.dtype == np.float32
    assert batch_bgr.min() >= 0.0
    assert batch_bgr.max() <= 1.0
    assert orig_w == 200
    assert orig_h == 150
    assert rgb_128.shape == (128, 128, 3)


def test_real_inference_on_sample_ultrasound():
    service = ModelService()
    sample_bytes = create_sample_ultrasound_image(300, 300)
    result = service.predict(sample_bytes)

    assert result["inference_mode"] == "real"
    assert "Residual CNN" in result["model_name"]
    
    # Verify 9 original class probabilities
    breakdown = result["original_class_breakdown"]
    assert len(breakdown) == 9
    prob_sum = sum(c["probability"] for c in breakdown)
    assert prob_sum == pytest.approx(1.0, rel=1e-3)
    
    for c in breakdown:
        assert np.isfinite(c["probability"])
        assert 0.0 <= c["probability"] <= 1.0

    # Verify explainability Grad-CAM output
    explain = result["explainability"]
    assert explain["available"] is True
    assert explain["original_image_url"].startswith("data:image/jpeg;base64,")
    assert explain["attribution_overlay_url"].startswith("data:image/jpeg;base64,")
    assert "attribution" in explain["notice"].lower()


# -------------------------------------------------------------
# 3. FastAPI Endpoint Tests
# -------------------------------------------------------------
def test_api_health():
    data = health()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["model_error"] is None


def test_api_model_info():
    data = model_info()
    assert data["model_loaded"] is True
    assert data["output_classes"] == 9
    assert len(data["candidate_labels"]) == 9
    assert "Tier_0" in data["tiers"]
    assert "Tier_1" in data["tiers"]
    assert "Tier_2" in data["tiers"]
    assert "Review_Required" in data["tiers"]


def test_api_predict_valid_jpeg():
    sample_bytes = create_sample_ultrasound_image(256, 256, format="JPEG")
    upload = UploadFile(
        filename="ultrasound_scan.jpg",
        file=io.BytesIO(sample_bytes),
        headers={"content-type": "image/jpeg"}
    )
    res = predict(upload)
    assert res.status_code == 200
    data = json.loads(res.body)
    assert data["inference_mode"] == "real"
    assert len(data["original_class_breakdown"]) == 9
    assert "dominant_original_prediction" in data
    assert "tier_scores" in data
    assert "explainability" in data


def test_api_predict_valid_png():
    sample_bytes = create_sample_ultrasound_image(128, 128, format="PNG")
    upload = UploadFile(
        filename="ultrasound_scan.png",
        file=io.BytesIO(sample_bytes),
        headers={"content-type": "image/png"}
    )
    res = predict(upload)
    assert res.status_code == 200
    data = json.loads(res.body)
    assert data["inference_mode"] == "real"


def test_api_predict_invalid_mime_type():
    upload = UploadFile(
        filename="report.pdf",
        file=io.BytesIO(b"%PDF-1.4 dummy"),
        headers={"content-type": "application/pdf"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict(upload)
    assert exc_info.value.status_code == 415
    assert "Unsupported file type" in exc_info.value.detail


def test_api_predict_corrupt_image():
    corrupted = b"NOT_AN_IMAGE_CONTENT" * 50
    upload = UploadFile(
        filename="corrupt.jpg",
        file=io.BytesIO(corrupted),
        headers={"content-type": "image/jpeg"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict(upload)
    assert exc_info.value.status_code == 422
    assert "Corrupt" in exc_info.value.detail or "Invalid" in exc_info.value.detail


def test_api_predict_image_too_small():
    tiny = Image.new("RGB", (16, 16), color="black")
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    upload = UploadFile(
        filename="tiny.png",
        file=io.BytesIO(buf.getvalue()),
        headers={"content-type": "image/png"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict(upload)
    assert exc_info.value.status_code == 422
    assert "too small" in exc_info.value.detail


def test_direct_python_vs_api_predict_consistency():
    """Verify that direct Python inference and FastAPI /predict produce identical numerical outputs."""
    sample_bytes = create_sample_ultrasound_image(128, 128, format="JPEG")
    service = ModelService()
    
    # 1. Direct Python inference
    direct_res = service.predict(sample_bytes)
    
    # 2. FastAPI endpoint inference
    upload = UploadFile(
        filename="scan.jpg",
        file=io.BytesIO(sample_bytes),
        headers={"content-type": "image/jpeg"}
    )
    api_res_obj = predict(upload)
    api_res = json.loads(api_res_obj.body)
    
    # Assert top prediction agreement
    assert direct_res["dominant_original_prediction"]["index"] == api_res["dominant_original_prediction"]["index"]
    assert direct_res["dominant_original_prediction"]["class_name"] == api_res["dominant_original_prediction"]["class_name"]
    assert direct_res["dominant_original_prediction"]["probability"] == pytest.approx(api_res["dominant_original_prediction"]["probability"], abs=1e-5)
    
    # Assert all 9 class probabilities match
    for d_item, a_item in zip(direct_res["original_class_breakdown"], api_res["original_class_breakdown"]):
        assert d_item["index"] == a_item["index"]
        assert d_item["class_name"] == a_item["class_name"]
        assert d_item["probability"] == pytest.approx(a_item["probability"], abs=1e-5)
        
    # Assert tier scores match
    for tier in ["Tier_0", "Tier_1", "Tier_2", "Review_Required"]:
        assert direct_res["tier_scores"][tier] == pytest.approx(api_res["tier_scores"][tier], abs=1e-5)


def test_verified_mapping_status():
    """Verify that mapping is verified and tier triage is active."""
    service = ModelService()
    assert service.mapping_verified is True
    data = health()
    assert data["mapping_verified"] is True
    assert data["mapping_status"] == "VERIFIED"
    
    # Check class mapping order
    assert service.labels[0] == "Various causes of gallbladder wall thickening"
    assert service.labels[1] == "Gallstones"
    assert service.labels[2] == "Abdomen and retroperitoneum"
    assert service.labels[3] == "Cholecystitis"
    assert service.labels[4] == "Membranous and gangrenous cholecystitis"
    assert service.labels[5] == "Perforation"
    assert service.labels[6] == "Polyps and cholesterol crystals"
    assert service.labels[7] == "Adenomyomatosis"
    assert service.labels[8] == "Carcinoma"


def test_real_dataset_gallstones_sample():
    """Test actual Gallstones image from dataset or sample images."""
    dataset_path = Path(r"C:\Users\Vyomesh\OneDrive\Desktop\Gallbladder_Project\original_dataset\1Gallstones")
    fallback_sample = Path(__file__).resolve().parent.parent / "external_validation" / "images" / "ext_gallstone_02.jpg"
    service = ModelService()
    
    img_bytes = None
    try:
        if dataset_path.exists():
            img_files = list(dataset_path.rglob("*.jpg"))
            if img_files:
                with open(img_files[0], "rb") as f:
                    img_bytes = f.read()
    except (PermissionError, OSError):
        pass

    if img_bytes is None and fallback_sample.exists():
        with open(fallback_sample, "rb") as f:
            img_bytes = f.read()

    if img_bytes is not None:
        res = service.predict(img_bytes)
        assert res["dominant_original_prediction"]["index"] == 1
        assert res["dominant_original_prediction"]["class_name"] == "Gallstones"
        assert res["assigned_triage_category"] == "Tier_0"
        assert res["triage_blocked"] is False


def test_confidence_reporting_and_disclaimer():
    """Verify confidence reporting, Model Softmax Score label, and clinical disclaimers."""
    from tier_mapping import format_softmax_score, format_tier_score, aggregate_triage
    
    # 1. Test near-100% value (e.g. 0.99995) does not misleadingly round to 100.00%
    assert format_softmax_score(0.99995) == ">99.99%"
    assert format_softmax_score(0.9999) == ">99.99%"
    assert format_softmax_score(1.0) == ">99.99%"
    
    # 2. Test near-zero value (e.g. 0.00003) does not misleadingly round to 0.00%
    assert format_softmax_score(0.00003) == "<0.01%"
    assert format_softmax_score(0.0001) == "<0.01%"
    
    # 3. Test standard percentage
    assert format_softmax_score(0.9474) == "94.74%"
    
    # 4. Test tier formatting
    assert format_tier_score(0.9995) == ">99.9%"
    assert format_tier_score(0.0003) == "<0.1%"
    assert format_tier_score(0.852) == "85.2%"

    # 5. Test aggregate_triage output structure
    probs = np.array([0.00001, 0.99995, 0.0, 0.0, 0.0, 0.00002, 0.0, 0.00002, 0.0])
    res = aggregate_triage(probs, mapping_verified=True)
    
    dom = res["dominant_original_prediction"]
    assert dom["score_type"] == "Model Softmax Score"
    assert dom["formatted_score"] == ">99.99%"
    assert dom["calibration_status"] == "uncalibrated"
    
    # Mathematical probability must remain unaltered
    assert pytest.approx(dom["probability"]) == 0.99995
    
    # Class breakdown must have formatted_score and score_type
    assert res["original_class_breakdown"][0]["formatted_score"] == "<0.01%"
    assert res["original_class_breakdown"][0]["score_type"] == "Model Softmax Score"
    
    # Disclaimer must explicitly contain the exact required notice
    assert "Research prototype only. Model scores do not represent clinical diagnostic certainty." in res["clinical_disclaimer"]
