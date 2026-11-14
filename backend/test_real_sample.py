import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from model_service import ModelService

print("Initializing ModelService...")
service = ModelService()
assert service.ready, f"Service failed to initialize: {service.error}"

sample_path = BACKEND_DIR / "samples" / "sample_ultrasound_intact.jpg"
print(f"Reading sample ultrasound from {sample_path}...")
with open(sample_path, "rb") as f:
    raw_bytes = f.read()

print(f"Running REAL model inference on {len(raw_bytes):,} bytes...")
res = service.predict(raw_bytes)

print("\n================== REAL INFERENCE RESULTS ==================")
print(f"Model Name:          {res['model_name']}")
print(f"Inference Mode:      {res['inference_mode']}")
print(f"Mapping Status:      {res['mapping_status']}")
print(f"Triage Blocked:      {res['triage_blocked']}")
print(f"Dominant Prediction: {res['dominant_original_prediction']['class_name']} ({res['dominant_original_prediction']['percentage']}%)")
print(f"Output Class Index:  #{res['dominant_original_prediction']['index']}")
print(f"Assigned Triage:     {res['assigned_triage_name']}")
print(f"Triage Rationale:    {res['triage_rationale']}")
print("\n--- Tier Scores ---")
for t, score in res['tier_scores'].items():
    print(f"  {t:16s}: {score*100:6.2f}%")

print("\n--- 9-Class Distribution ---")
for c in res['original_class_breakdown']:
    print(f"  #{c['index']} {c['class_name']:46s}: {c['percentage']:6.2f}% -> {c['tier']}")

explain = res['explainability']
print(f"\n--- Explainability (Grad-CAM) ---")
print(f"  Available:         {explain['available']}")
print(f"  Target Class:      {explain['target_class']}")
print(f"  Layer Name:        {explain['layer_name']}")
print(f"  Overlay URL Len:   {len(explain['attribution_overlay_url'])} chars")
print(f"  Original URL Len:  {len(explain['original_image_url'])} chars")
print("============================================================\n")

