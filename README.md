# GallNet Triage — Automated Gallbladder Perforation Risk Research Platform

GallNet Triage is a specialized medical AI research web application designed to evaluate gallbladder ultrasound scans using an existing pretrained **Self-Attention-Guided Residual CNN** (Rashid et al., 2025). The platform executes **real neural inference** with the 9-class multiclass checkpoint and translates outputs into a provisional, three-tier gallbladder perforation triage system.

> **ACADEMIC & RESEARCH PROTOTYPE ONLY**  
> This software is intended strictly for academic investigation and machine learning benchmarking. It is **not FDA or CE cleared** for clinical diagnosis, real-time surgical decision-making, or automated patient triage. Triage scores represent post-hoc heuristic aggregations of multiclass probabilities, not calibrated clinical risk models.

---

## 1. System Architecture & Model Audit

### Pretrained Checkpoint
* **Checkpoint File**: `models/model-031-0.979685-0.992509.keras` (45.3 MB, Keras 3 format).
* **Architecture**: Self-Attention-Guided Residual CNN with dilated convolution branches, dot-product attention (`complex_attention`), depthwise separable convolutions, and dense classification layers.
* **Input Shape**: `(None, 128, 128, 3)`.
* **Output Shape**: `(None, 9)` with softmax probability distribution.
* **Explainability Layer**: `conv2d_11` (the final convolutional block prior to global average pooling, output shape: `(None, 64, 64, 512)`).

### Image Preprocessing Pipeline
Directly audited from `gallbladder_ultrasound_self_attention.ipynb` (Cells 2, 4, 6):
1. **Color Order**: OpenCV BGR format.
2. **Spatial Resizing**: Bilinear interpolation to `128 x 128` pixels.
3. **Normalization**: Pixel values scaled to `[0.0, 1.0]` by dividing by `255.0`.

### Mapping Verification Status: **UNVERIFIED CANDIDATE MAPPING**
* **Audit Finding**: The reference training notebook loaded images from subdirectories named `['0', '1', '2', '3', '4', '5', '6', '7', '8']` and assigned one-hot vectors via `to_categorical(label, 9)`. The notebook contains **no documentation, comments, or dictionary linking directory numbers `0`–`8` to the original UlGBDatabase clinical classes `1`–`9`**.
* **Candidate Hypothesis**: Directory index $i$ corresponds to UlGBDatabase folder $(i+1)$.
* **Safety & Transparency Policy**: The application marks the mapping status as **UNVERIFIED** (`backend/class_mapping.json`). While the system reports genuine, real 9-class model predictions, the derived 3-tier surgical triage assignment is flagged as **PROVISIONAL / BLOCKED** until explicit verification via labeled validation scans or author confirmation.

---

## 2. Provisional Three-Tier Triage Aggregation

When candidate ordering is applied, the 9 original disease classes map to three provisional research tiers and a clinical review category:

| Triage Tier | Disease Category | Candidate Index | Description |
| :--- | :--- | :---: | :--- |
| **Tier 0** (Intact Wall) | Gallstones<br>Cholecystitis<br>Polyps & cholesterol crystals<br>Adenomyomatosis<br>Gallbladder wall thickening | `0`<br>`2`<br>`5`<br>`6`<br>`8` | Structurally preserved gallbladder wall; minimal immediate risk of transmural wall breakdown. |
| **Tier 1** (Compromised Wall / Pre-perforation) | Membranous and gangrenous cholecystitis | `3` | Severe transmural inflammation, ischemia, or acute necrosis with heightened perforation hazard. |
| **Tier 2** (Perforated) | Perforation | `4` | Frank gallbladder wall disruption or macroscopic perforation. |
| **Review Required** | Abdomen and retroperitoneum<br>Carcinoma | `1`<br>`7` | Non-gallbladder abdominal pathology or malignancy requiring specialized oncological / surgical review. |

### Mathematical Aggregation Rules:
* $\text{Score}_{\text{Tier 0}} = p_0 + p_2 + p_5 + p_6 + p_8$
* $\text{Score}_{\text{Tier 1}} = p_3$
* $\text{Score}_{\text{Tier 2}} = p_4$
* $\text{Score}_{\text{Review}} = p_1 + p_7$
* The four categories sum to $1.0$ without discarding or artificially renormalizing review probability.
* If the dominant original prediction is Carcinoma or Abdomen, the triage state is forced to **REVIEW REQUIRED**.

---

## 3. Windows Setup & Quick Start

### Prerequisites
* Windows 10/11
* Python 3.11 or 3.12 (via [python.org](https://www.python.org/))
* Optional: Node.js 18+ (for frontend Vite hot-reload development)

### Quick Start (Double-Click Batch Scripts)

1. **Run Setup**:
   Double-click `setup_windows.bat`. This automatically:
   - Configures the Python virtual environment (`backend\.venv`).
   - Installs dependencies from `backend\requirements.txt`.
   - Confirms the presence of `models\model-031-0.979685-0.992509.keras`.

2. **Start Backend**:
   Double-click `start_backend_windows.bat`.
   - Backend URL: `http://127.0.0.1:8000`
   - Research Web Dashboard: `http://127.0.0.1:8000/`
   - Health Probe: `http://127.0.0.1:8000/health`
   - Interactive API Docs: `http://127.0.0.1:8000/docs`

3. **Start Frontend (Optional Vite Dev Server)**:
   Double-click `start_frontend_windows.bat`.
   - Starts Vite development server at `http://localhost:5173`.
   - *Note: If Node.js is not installed, the full interactive application is already served directly by the FastAPI backend at `http://127.0.0.1:8000/`.*

---

## 4. Manual PowerShell / Command Prompt Instructions

```powershell
# Navigate to project root
cd c:\Users\Vyomesh\OneDrive\Desktop\gallnet-tier-app

# 1. Run Backend Unit and Integration Tests
py -3.12 -m pytest backend/ -v

# 2. Run Backend with Uvicorn
cd backend
py -3.12 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 5. API Reference

### `GET /health`
Returns model readiness status and verification metadata.
```json
{
  "status": "online",
  "model_loaded": true,
  "model_name": "Self-Attention-Guided Residual CNN (Rashid et al., 2025)",
  "model_error": null,
  "mapping_verified": false,
  "mapping_status": "UNVERIFIED"
}
```

### `GET /model-info`
Returns input resolution, architecture layer names, candidate class list, and tier grouping definitions.

### `POST /predict`
Accepts multipart form-data (`file`: PNG or JPEG).
Returns:
* `dominant_original_prediction`: Class name, output index, real probability.
* `original_class_breakdown`: List of all 9 original class probabilities.
* `tier_scores`: Aggregated scores for Tier 0, Tier 1, Tier 2, and Review Required.
* `mapping_status`: `"UNVERIFIED"` (blocks automatic triage).
* `explainability`: Base64 JPEG data URL of the Grad-CAM activation heatmap overlay on layer `conv2d_11`.

---

## 6. Verification and Testing Summary

The project includes an automated test suite (`backend/test_app.py` & `backend/test_mapping.py`) covering:
* Pretrained Keras checkpoint loading and layer verification.
* BGR color channel order preservation.
* Input shape `(1, 128, 128, 3)` and output shape `(1, 9)`.
* Mathematical validation of Tier 0, Tier 1, Tier 2, and Review Required sums.
* Validation of `Review Required` triggered by Carcinoma or abdominal pathology.
* Grad-CAM gradient tape attribution generation.
* Real inference execution with synthetic ultrasound phantom scans.
* Input validation (mime-type rejection, corruption handling, min-size enforcement).

To execute all tests:
```cmd
python -m pytest backend/ -v
```

---

## 7. License & Credits

* Reference Pretrained Weights & Architecture: Rashid et al., 2025 / [gallbladder_ultrasound](https://github.com/rakeshchandrajoshi/gallbladder_ultrasound)
* Ultrasound Dataset: UlGBDatabase
* Derived Triage Framework: Provisional 3-tier research grouping for academic investigation of perforation risk indicators.
