"""Three-Tier Surgical Risk Triage Aggregator for Gallbladder Ultrasound.

Translates the 9-class multiclass model output into a provisional
three-tier gallbladder perforation risk research triage scheme:
  - Tier 0 (Intact Wall): Gallstones, Cholecystitis, Polyps & cholesterol crystals,
                          Adenomyomatosis, Wall thickening
  - Tier 1 (Compromised Wall / Pre-perforation): Membranous and gangrenous cholecystitis
  - Tier 2 (Perforated): Perforation
  - Review Required: Abdomen and retroperitoneum, Carcinoma

IMPORTANT: This is a post-hoc aggregation of a multiclass classifier,
NOT a newly trained ordinal deep learning model or calibrated clinical risk model.
"""

from typing import Dict, List, Any, Optional
import numpy as np

TIER_MEMBERS = {
    "Tier_0": {
        "name": "Tier 0 — Intact Wall",
        "classes": {
            "Gallstones",
            "Cholecystitis",
            "Polyps and cholesterol crystals",
            "Adenomyomatosis",
            "Various causes of gallbladder wall thickening"
        },
        "description": "Gallbladder wall remains structurally intact; low immediate perforation risk."
    },
    "Tier_1": {
        "name": "Tier 1 — Compromised Wall / Pre-perforation",
        "classes": {
            "Membranous and gangrenous cholecystitis"
        },
        "description": "Severe transmural inflammation or ischemia presenting heightened perforation risk."
    },
    "Tier_2": {
        "name": "Tier 2 — Perforated",
        "classes": {
            "Perforation"
        },
        "description": "Frank gallbladder wall disruption or frank perforation."
    },
    "Review_Required": {
        "name": "Review Required",
        "classes": {
            "Abdomen and retroperitoneum",
            "Carcinoma"
        },
        "description": "Pathology outside benign/inflammatory progression spectrum requiring specialized oncological or anatomical surgical review."
    }
}

VERIFIED_ORDER = [
    "Various causes of gallbladder wall thickening",    # 0 -> Tier 0 (Folder 9 % 9 = 0)
    "Gallstones",                                       # 1 -> Tier 0 (Folder 1)
    "Abdomen and retroperitoneum",                      # 2 -> Review Required (Folder 2)
    "Cholecystitis",                                    # 3 -> Tier 0 (Folder 3)
    "Membranous and gangrenous cholecystitis",          # 4 -> Tier 1 (Folder 4)
    "Perforation",                                      # 5 -> Tier 2 (Folder 5)
    "Polyps and cholesterol crystals",                  # 6 -> Tier 0 (Folder 6)
    "Adenomyomatosis",                                  # 7 -> Tier 0 (Folder 7)
    "Carcinoma"                                         # 8 -> Review Required (Folder 8)
]

# Kept for backward compatibility
CANDIDATE_ORDER = VERIFIED_ORDER

ALIASES = {
    "various causes of gallbladder wall thickening": "Various causes of gallbladder wall thickening",
    "gallbladder wall thickening": "Various causes of gallbladder wall thickening",
    "wall thickening": "Various causes of gallbladder wall thickening",
    "polyps and cholesterol crystal": "Polyps and cholesterol crystals",
    "polyps and cholesterol crystals": "Polyps and cholesterol crystals",
    "membranous/gangrenous cholecystitis": "Membranous and gangrenous cholecystitis",
    "membranous and gangrenous cholecystitis": "Membranous and gangrenous cholecystitis",
    "gangrenous cholecystitis": "Membranous and gangrenous cholecystitis",
    "abdomen and retroperitoneal pathology": "Abdomen and retroperitoneum",
    "intraabdominal and retroperitoneum problems": "Abdomen and retroperitoneum",
    "abdomen and retroperitoneum": "Abdomen and retroperitoneum",
    "perforation of gb": "Perforation",
    "perforation": "Perforation",
    "adenomyomatosis of the gb": "Adenomyomatosis",
    "adenomyomatosis": "Adenomyomatosis",
    "carcinoma": "Carcinoma",
    "gallbladder cancer": "Carcinoma",
    "gallstones": "Gallstones",
    "cholecystitis": "Cholecystitis"
}

ALL_CANONICAL_CLASSES = set().union(*(v["classes"] for v in TIER_MEMBERS.values()))


def format_softmax_score(prob: float) -> str:
    """
    Formats a softmax probability score accurately without misleading 100.00% rounding:
    - Values >= 0.9999 (e.g. 0.99995) are displayed as '>99.99%' to prevent false certainty.
    - Values > 0.0 and <= 0.0001 are displayed as '<0.01%'.
    - All other values are formatted to 2 decimal places (e.g. '94.74%').
    """
    val = float(prob)
    if val >= 0.9999:
        return ">99.99%"
    elif 0.0 < val <= 0.0001:
        return "<0.01%"
    else:
        return f"{val * 100:.2f}%"


def format_tier_score(prob: float) -> str:
    """
    Formats an aggregated tier score without misleading 100.0% rounding:
    - Values >= 0.999 are displayed as '>99.9%'
    - Values > 0.0 and <= 0.001 are displayed as '<0.1%'
    - Other values are formatted to 1 decimal place (e.g. '78.8%')
    """
    val = float(prob)
    if val >= 0.999:
        return ">99.9%"
    elif 0.0 < val <= 0.001:
        return "<0.1%"
    else:
        return f"{val * 100:.1f}%"


def normalize_class_name(name: str) -> str:
    cleaned = " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())
    return ALIASES.get(cleaned, str(name).strip())


def validate_labels(labels: List[str]):
    if len(labels) != 9:
        raise ValueError(f"Expected 9 class labels, got {len(labels)}")
    canonical = [normalize_class_name(x) for x in labels]
    if len(set(canonical)) != 9:
        raise ValueError("All 9 class labels must be distinct")
    unknown = set(canonical) - ALL_CANONICAL_CLASSES
    if unknown:
        raise ValueError(f"Unrecognized class labels: {unknown}")


def aggregate_triage(
    probabilities: np.ndarray,
    labels: Optional[List[str]] = None,
    mapping_verified: bool = False
) -> Dict[str, Any]:
    """
    Aggregates 9-class probabilities into Tier 0, Tier 1, Tier 2, and Review Required.
    """
    raw = np.asarray(probabilities, dtype=np.float64).flatten()
    if raw.shape != (9,):
        raise ValueError(f"Expected 9 class scores, got shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Scores must all be finite numbers")

    class_names = labels if (labels and len(labels) == 9) else CANDIDATE_ORDER
    validate_labels(class_names)
    canonical_labels = [normalize_class_name(x) for x in class_names]

    # Convert logits to probabilities via softmax if values are out of [0, 1] or don't sum near 1.0
    if (raw < 0.0).any() or (raw > 1.0).any() or not np.isclose(raw.sum(), 1.0, atol=1e-2):
        shifted = raw - np.max(raw)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)
    else:
        total = raw.sum()
        if total <= 0:
            raise ValueError("Sum of probabilities must be positive")
        probs = raw / total

    # Aggregate tier scores based on canonical class names
    tier_scores = {tier_key: 0.0 for tier_key in TIER_MEMBERS.keys()}
    for i, cname in enumerate(canonical_labels):
        for tier_key, meta in TIER_MEMBERS.items():
            if cname in meta["classes"]:
                tier_scores[tier_key] += float(probs[i])
                break

    dominant_idx = int(np.argmax(probs))
    dominant_class = class_names[dominant_idx]
    dominant_canonical = canonical_labels[dominant_idx]
    dominant_prob = float(probs[dominant_idx])

    # Determine assigned dominant triage category
    if dominant_canonical in TIER_MEMBERS["Review_Required"]["classes"]:
        assigned_tier = "Review_Required"
        triage_rationale = (
            f"Dominant predicted class is '{dominant_class}', which represents an excluded disease "
            "category (malignancy or non-gallbladder abdominal pathology) requiring specialized clinical review."
        )
    else:
        primary_tiers = {
            "Tier_0": tier_scores["Tier_0"],
            "Tier_1": tier_scores["Tier_1"],
            "Tier_2": tier_scores["Tier_2"]
        }
        assigned_tier = max(primary_tiers, key=primary_tiers.get)
        triage_rationale = (
            f"Aggregated score for {assigned_tier} is {format_tier_score(tier_scores[assigned_tier])}. "
            f"Dominant original prediction is '{dominant_class}' ({format_softmax_score(dominant_prob)})."
        )

    class_breakdown = [
        {
            "index": i,
            "class_name": class_names[i],
            "probability": float(probs[i]),
            "percentage": round(float(probs[i]) * 100, 2),
            "formatted_score": format_softmax_score(float(probs[i])),
            "score_type": "Model Softmax Score",
            "tier": (
                "Tier_0" if canonical_labels[i] in TIER_MEMBERS["Tier_0"]["classes"]
                else "Tier_1" if canonical_labels[i] in TIER_MEMBERS["Tier_1"]["classes"]
                else "Tier_2" if canonical_labels[i] in TIER_MEMBERS["Tier_2"]["classes"]
                else "Review_Required"
            )
        }
        for i in range(9)
    ]

    status_code = "VERIFIED" if mapping_verified else "UNVERIFIED"
    verification_warning = (
        None if mapping_verified else
        "CAUTION: Model output index mapping is UNVERIFIED. The training code indexed directories 0–8 "
        "without documented class mapping. The three-tier triage values shown are candidate research "
        "aggregations and must not be used for surgical or clinical triage without validation."
    )

    formatted_tier_scores = {
        k: format_tier_score(v) for k, v in tier_scores.items()
    }

    return {
        "status": "unsupported_category" if assigned_tier == "Review_Required" else "success",
        "predicted_tier": assigned_tier if mapping_verified else None,
        "raw_tier_probabilities": tier_scores,
        "formatted_tier_scores": formatted_tier_scores,
        "mapping_status": status_code,
        "mapping_verified": mapping_verified,
        "verification_warning": verification_warning,
        "triage_blocked": not mapping_verified,
        "dominant_original_prediction": {
            "index": dominant_idx,
            "class_name": dominant_class,
            "probability": dominant_prob,
            "percentage": round(dominant_prob * 100, 2),
            "formatted_score": format_softmax_score(dominant_prob),
            "score_type": "Model Softmax Score",
            "calibration_status": "uncalibrated"
        },
        "assigned_triage_category": assigned_tier,
        "assigned_triage_name": TIER_MEMBERS[assigned_tier]["name"],
        "triage_rationale": triage_rationale,
        "tier_scores": tier_scores,
        "tier_definitions": {k: v["description"] for k, v in TIER_MEMBERS.items()},
        "original_class_breakdown": class_breakdown,
        "is_post_hoc_aggregation": True,
        "clinical_disclaimer": (
            "Research prototype only. Model scores do not represent clinical diagnostic certainty. "
            "Triage scores are post-hoc mathematical aggregations of an uncalibrated 9-class deep neural "
            "network and do NOT represent calibrated clinical risk or surgical urgency timing."
        )
    }


def aggregate(scores, labels=None, threshold=0.5, mapping_verified=True):
    return aggregate_triage(scores, labels=labels, mapping_verified=mapping_verified)
