import numpy as np
import pytest
from tier_mapping import aggregate
LABELS = ["gallstones", "cholecystitis", "polyps and cholesterol crystals", "adenomyomatosis", "wall thickening", "membranous and gangrenous cholecystitis", "perforation", "carcinoma", "abdomen and retroperitoneum"]

def test_tier_0_sum():
    result = aggregate([.1,.1,.1,.1,.1,.2,.2,.05,.05], LABELS)
    assert result["predicted_tier"] == "Tier_0"
    assert result["raw_tier_probabilities"]["Tier_0"] == pytest.approx(.5)

def test_tier_2():
    assert aggregate([.01]*6 + [.90,.02,.02], LABELS)["predicted_tier"] == "Tier_2"

def test_excluded():
    assert aggregate([.02]*7 + [.80,.06], LABELS)["status"] == "unsupported_category"

def test_logits():
    result = aggregate(np.array([0,0,0,0,0,0,10,0,0]), LABELS)
    assert result["predicted_tier"] == "Tier_2"

def test_invalid_labels():
    with pytest.raises(ValueError):
        aggregate([1/9]*9, ["x"]*9)
