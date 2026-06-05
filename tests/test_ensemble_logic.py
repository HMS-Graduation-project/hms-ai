"""
Unit tests for ensemble calculation logic.
No model loading required — tests pure math and agreement functions.
"""
import pytest


# ── Weights ─────────────────────────────────────────────────────────

WEIGHTS = {"DenseNet121": 0.40, "EfficientNet-B0": 0.35, "ResNet50": 0.25}
THRESHOLD = 0.94


def weighted_average(probs: dict[str, float]) -> float:
    return round(sum(probs[k] * WEIGHTS[k] for k in WEIGHTS), 4)


def risk_level(prob: float) -> str:
    if prob >= THRESHOLD:
        return "HIGH"
    elif prob >= 0.70:
        return "ELEVATED"
    elif prob >= 0.30:
        return "MODERATE"
    else:
        return "LOW"


def model_agreement(predictions: list[str]) -> tuple[str, float]:
    pneumonia = predictions.count("PNEUMONIA")
    normal = predictions.count("NORMAL")
    mx = max(pneumonia, normal)
    if mx == 3:
        return "STRONG", 1.0
    elif mx == 2:
        return "MODERATE", 0.67
    else:
        return "LOW", 0.33


# ── Weight Tests ────────────────────────────────────────────────────

class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_densenet_has_highest_weight(self):
        assert WEIGHTS["DenseNet121"] > WEIGHTS["EfficientNet-B0"]
        assert WEIGHTS["DenseNet121"] > WEIGHTS["ResNet50"]


# ── Weighted Average Tests ──────────────────────────────────────────

class TestWeightedAverage:
    def test_example_from_spec(self):
        """Example: 0.72*0.40 + 0.81*0.35 + 0.75*0.25 = 0.7595"""
        probs = {"DenseNet121": 0.72, "EfficientNet-B0": 0.81, "ResNet50": 0.75}
        result = weighted_average(probs)
        assert abs(result - 0.7595) < 0.001

    def test_all_zero(self):
        probs = {"DenseNet121": 0.0, "EfficientNet-B0": 0.0, "ResNet50": 0.0}
        assert weighted_average(probs) == 0.0

    def test_all_one(self):
        probs = {"DenseNet121": 1.0, "EfficientNet-B0": 1.0, "ResNet50": 1.0}
        assert abs(weighted_average(probs) - 1.0) < 0.001

    def test_high_densenet_low_others(self):
        probs = {"DenseNet121": 0.99, "EfficientNet-B0": 0.10, "ResNet50": 0.10}
        result = weighted_average(probs)
        # 0.99*0.40 + 0.10*0.35 + 0.10*0.25 = 0.396 + 0.035 + 0.025 = 0.456
        assert abs(result - 0.456) < 0.001

    def test_threshold_boundary(self):
        """Exactly at threshold should be HIGH"""
        probs = {"DenseNet121": 0.94, "EfficientNet-B0": 0.94, "ResNet50": 0.94}
        result = weighted_average(probs)
        assert abs(result - 0.94) < 0.001
        assert risk_level(result) == "HIGH"


# ── Risk Level Tests ────────────────────────────────────────────────

class TestRiskLevel:
    def test_low(self):
        assert risk_level(0.10) == "LOW"
        assert risk_level(0.0) == "LOW"
        assert risk_level(0.29) == "LOW"

    def test_moderate(self):
        assert risk_level(0.30) == "MODERATE"
        assert risk_level(0.50) == "MODERATE"
        assert risk_level(0.69) == "MODERATE"

    def test_elevated(self):
        assert risk_level(0.70) == "ELEVATED"
        assert risk_level(0.80) == "ELEVATED"
        assert risk_level(0.93) == "ELEVATED"

    def test_high(self):
        assert risk_level(0.94) == "HIGH"
        assert risk_level(0.99) == "HIGH"
        assert risk_level(1.0) == "HIGH"


# ── Model Agreement Tests ──────────────────────────────────────────

class TestModelAgreement:
    def test_strong_all_normal(self):
        agreement, score = model_agreement(["NORMAL", "NORMAL", "NORMAL"])
        assert agreement == "STRONG"
        assert score == 1.0

    def test_strong_all_pneumonia(self):
        agreement, score = model_agreement(["PNEUMONIA", "PNEUMONIA", "PNEUMONIA"])
        assert agreement == "STRONG"
        assert score == 1.0

    def test_moderate_two_normal(self):
        agreement, score = model_agreement(["NORMAL", "NORMAL", "PNEUMONIA"])
        assert agreement == "MODERATE"
        assert score == 0.67

    def test_moderate_two_pneumonia(self):
        agreement, score = model_agreement(["PNEUMONIA", "PNEUMONIA", "NORMAL"])
        assert agreement == "MODERATE"
        assert score == 0.67

    def test_agreement_score_values(self):
        _, strong = model_agreement(["NORMAL", "NORMAL", "NORMAL"])
        _, moderate = model_agreement(["NORMAL", "NORMAL", "PNEUMONIA"])
        assert strong > moderate


# ── Integration: Full Ensemble Decision ─────────────────────────────

class TestEnsembleDecision:
    def test_below_threshold_is_normal(self):
        probs = {"DenseNet121": 0.72, "EfficientNet-B0": 0.81, "ResNet50": 0.75}
        final = weighted_average(probs)
        assert final < THRESHOLD
        assert risk_level(final) == "ELEVATED"

    def test_above_threshold_is_pneumonia(self):
        probs = {"DenseNet121": 0.96, "EfficientNet-B0": 0.95, "ResNet50": 0.94}
        final = weighted_average(probs)
        assert final >= THRESHOLD
        assert risk_level(final) == "HIGH"

    def test_moderate_probability(self):
        probs = {"DenseNet121": 0.50, "EfficientNet-B0": 0.45, "ResNet50": 0.40}
        final = weighted_average(probs)
        assert risk_level(final) == "MODERATE"
