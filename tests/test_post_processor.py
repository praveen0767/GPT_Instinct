"""
tests/test_post_processor.py
Unit tests for the OCR post-processor module.
Covers: calibration, domain validation, decimal analysis, flags, and spec examples.
"""
import math
import pytest
from ocr_pipeline.llm_post_processor import (
    OCRPostProcessor,
    LOGISTIC_STEEPNESS, LOGISTIC_CENTER,
    CANDIDATE_WEIGHT, IQ_WEIGHT,
    DOMAIN_VIOLATION_MULT, QC_THRESHOLD,
)


@pytest.fixture
def processor():
    return OCRPostProcessor()


# ── 1. Deterministic calibration ─────────────────────────────────────────────

class TestCalibration:
    def test_high_confidence_high_iq(self, processor):
        """High OCR confidence + good image quality → high calibrated conf."""
        iq = {"brightness": 0.8, "sharpness": 0.9, "glare": 0.1, "tilt_deg": 0}
        iq_factor = processor._compute_iq_factor(iq)
        cal = processor._calibrate(0.95, iq_factor)
        assert cal > 0.90, f"Expected >0.90, got {cal:.4f}"

    def test_low_confidence_low_iq(self, processor):
        """Low OCR confidence + poor image quality → low calibrated conf."""
        iq = {"brightness": 0.2, "sharpness": 0.3, "glare": 0.7, "tilt_deg": 15}
        iq_factor = processor._compute_iq_factor(iq)
        cal = processor._calibrate(0.3, iq_factor)
        assert cal < 0.5, f"Expected <0.50, got {cal:.4f}"

    def test_logistic_center_gives_0_5(self, processor):
        """raw_prob at 0.5 should give calibrated ~0.5 (sigmoid center)."""
        cal = processor._calibrate(0.5, 0.5)  # raw_prob = 0.7*0.5 + 0.3*0.5 = 0.5
        assert abs(cal - 0.5) < 0.01, f"Expected ~0.5, got {cal:.4f}"

    def test_iq_factor_computation(self, processor):
        """mean(brightness, sharpness) * (1 - glare)."""
        iq = {"brightness": 0.8, "sharpness": 0.6, "glare": 0.2}
        factor = processor._compute_iq_factor(iq)
        expected = ((0.8 + 0.6) / 2.0) * (1.0 - 0.2)  # 0.7 * 0.8 = 0.56
        assert abs(factor - expected) < 0.001

    def test_historical_calibration(self, processor):
        """When historical logistic params are provided, use them."""
        hist = {"logistic_a": -4.0, "logistic_b": 5.2}
        cal = processor._calibrate(0.9, 0.8, historical=hist)
        # 1 / (1 + exp(-(-4 + 5.2 * raw_prob)))
        raw_prob = 0.9 * CANDIDATE_WEIGHT + 0.8 * IQ_WEIGHT
        expected = 1.0 / (1.0 + math.exp(-(-4.0 + 5.2 * raw_prob)))
        assert abs(cal - expected) < 0.001


# ── 2. Domain validation ─────────────────────────────────────────────────────

class TestDomainValidation:
    def test_kwh_in_range(self, processor):
        """kWh value within range should not flag OUT_OF_RANGE."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {"value": "12345.6", "probability": 0.9, "candidates": []}},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "OUT_OF_RANGE" not in kwh_r["flags"]

    def test_kwh_out_of_range(self, processor):
        """kWh value exceeding max should flag OUT_OF_RANGE."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {"value": "99999999", "probability": 0.9, "candidates": []}},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "OUT_OF_RANGE" in kwh_r["flags"]
        assert kwh_r["confidence"] < 0.5  # should be penalized

    def test_serial_pattern_match(self, processor):
        """Serial matching pattern should not flag NONMATCH_SERIAL."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"meter_serial": {"value": "AB12-3456", "probability": 0.9, "candidates": []}},
        )
        serial_r = next(r for r in result["results"] if r["field"] == "serial")
        assert "NONMATCH_SERIAL" not in serial_r["flags"]

    def test_serial_pattern_mismatch(self, processor):
        """Serial not matching pattern should flag NONMATCH_SERIAL."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"meter_serial": {"value": "ab", "probability": 0.9, "candidates": []}},
        )
        serial_r = next(r for r in result["results"] if r["field"] == "serial")
        assert "NONMATCH_SERIAL" in serial_r["flags"]


# ── 3. Decimal analysis ──────────────────────────────────────────────────────

class TestDecimalAnalysis:
    def test_conflicting_decimal_positions(self, processor):
        """Candidates with different dot positions → UNCERT_DECIMAL."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {
                "value": "123456",
                "probability": 0.5,
                "candidates": [
                    {"value": "123456", "score": 0.55},
                    {"value": "1234.56", "score": 0.35},
                ],
            }},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "UNCERT_DECIMAL" in kwh_r["flags"]

    def test_no_decimal_ambiguity(self, processor):
        """All candidates agree on decimal → no UNCERT_DECIMAL."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {
                "value": "12345.6",
                "probability": 0.9,
                "candidates": [
                    {"value": "12345.6", "score": 0.9},
                ],
            }},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "UNCERT_DECIMAL" not in kwh_r["flags"]

    def test_decimal_position_calculation(self, processor):
        """Verify decimal_position is computed correctly."""
        pos = processor._get_decimal_position("12345.6")
        assert pos == 1
        pos2 = processor._get_decimal_position("123.45")
        assert pos2 == 2
        pos3 = processor._get_decimal_position("12345")
        assert pos3 == 0


# ── 4. Character correction ──────────────────────────────────────────────────

class TestCharacterCorrection:
    def test_o_to_zero(self, processor):
        text, corrections = processor._apply_char_corrections("12O45")
        assert text == "12045"
        assert any("O→0" in c for c in corrections)

    def test_i_to_one(self, processor):
        text, corrections = processor._apply_char_corrections("I234")
        assert text == "1234"
        assert any("I→1" in c for c in corrections)

    def test_s_to_five(self, processor):
        text, corrections = processor._apply_char_corrections("S678")
        assert text == "5678"

    def test_multiple_subs(self, processor):
        text, corrections = processor._apply_char_corrections("OI.S")
        assert text == "01.5"
        assert len(corrections) == 3

    def test_no_corrections_needed(self, processor):
        text, corrections = processor._apply_char_corrections("12345.6")
        assert text == "12345.6"
        assert len(corrections) == 0


# ── 5. Flag generation ───────────────────────────────────────────────────────

class TestFlagGeneration:
    def test_low_confidence_flag(self, processor):
        """Low calibrated conf → LOW_CONFIDENCE flag."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {"value": "12345", "probability": 0.2, "candidates": []}},
            image_quality={"blur": True, "glare": True, "tilt_deg": 0, "not_legible": False},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "LOW_CONFIDENCE" in kwh_r["flags"]

    def test_image_quality_poor(self, processor):
        """Not-legible image → overall_pass=false, IMAGE_QUALITY_POOR."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {"value": "12345", "probability": 0.9, "candidates": []}},
            image_quality={"blur": True, "glare": True, "tilt_deg": 50, "not_legible": True},
        )
        assert result["overall_pass"] is False
        assert "IMAGE_QUALITY_POOR" in result["qc_reasons"]

    def test_missing_digits(self, processor):
        """Empty value → MISSING_DIGITS flag."""
        result = processor.process(
            image_id="test",
            raw_ocr_fields={"kwh": {"value": "—", "probability": 0.0, "candidates": []}},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "MISSING_DIGITS" in kwh_r["flags"]


# ── 6. Spec example 1: decimal misplacement ──────────────────────────────────

class TestSpecExamples:
    def test_example1_decimal_correct(self, processor):
        """Spec example 1: '12345.6' with conf 0.60, good image quality."""
        result = processor.process(
            image_id="example1",
            raw_ocr_fields={"kwh": {
                "value": "12345.6",
                "probability": 0.60,
                "candidates": [
                    {"value": "12345.6", "score": 0.60},
                    {"value": "123456", "score": 0.30},
                    {"value": "12345.0", "score": 0.10},
                ],
            }},
            image_quality={"blur": False, "glare": False, "tilt_deg": 0, "not_legible": False},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert kwh_r["corrected"] == "12345.6"
        assert kwh_r["normalized_value"] == 12345.6
        assert kwh_r["decimal_position"] == 1
        assert kwh_r["unit"] == "kWh"

    def test_example2_ambiguous_decimal(self, processor):
        """Spec example 2: ambiguous decimal between '123456' and '1234.56'."""
        result = processor.process(
            image_id="example2",
            raw_ocr_fields={"kwh": {
                "value": "123456",
                "probability": 0.55,
                "candidates": [
                    {"value": "123456", "score": 0.55},
                    {"value": "1234.56", "score": 0.35},
                ],
            }},
            image_quality={"blur": False, "glare": True, "tilt_deg": 0, "not_legible": False},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert "UNCERT_DECIMAL" in kwh_r["flags"]
        # Should still produce a result
        assert kwh_r["corrected"] is not None

    def test_example3_serial_ocr_noise(self, processor):
        """Spec example 3: serial 'AB12-3456' with OCR noise."""
        result = processor.process(
            image_id="example3",
            raw_ocr_fields={"meter_serial": {
                "value": "AB12-3456",
                "probability": 0.88,
                "candidates": [
                    {"value": "AB12-3456", "score": 0.88},
                    {"value": "ABI2-3456", "score": 0.12},
                ],
            }},
        )
        serial_r = next(r for r in result["results"] if r["field"] == "serial")
        assert serial_r["corrected"] == "AB12-3456"
        assert serial_r["confidence"] > 0.7
        assert "NONMATCH_SERIAL" not in serial_r["flags"]


# ── 7. Full output structure ─────────────────────────────────────────────────

class TestOutputStructure:
    def test_output_has_required_keys(self, processor):
        result = processor.process(
            image_id="struct_test",
            raw_ocr_fields={"kwh": {"value": "999", "probability": 0.5, "candidates": []}},
        )
        assert "image_id" in result
        assert "results" in result
        assert "overall_pass" in result
        assert "qc_reasons" in result

    def test_field_result_has_all_keys(self, processor):
        result = processor.process(
            image_id="struct_test",
            raw_ocr_fields={"kwh": {"value": "999", "probability": 0.5, "candidates": []}},
        )
        field_r = result["results"][0]
        required_keys = [
            "field", "raw_best", "corrected", "normalized_value",
            "unit", "confidence", "decimal_confidence", "decimal_position",
            "flags", "reason",
        ]
        for key in required_keys:
            assert key in field_r, f"Missing key '{key}' in field result"

    def test_empty_input_fields(self, processor):
        """Processing with no fields should return empty results."""
        result = processor.process(
            image_id="empty",
            raw_ocr_fields={},
        )
        assert result["results"] == []
        assert result["overall_pass"] is True  # no failures

    def test_reason_under_120_chars(self, processor):
        """Reason string should never exceed 120 characters."""
        result = processor.process(
            image_id="reason_test",
            raw_ocr_fields={"kwh": {
                "value": "OISBZGT",  # many corrections
                "probability": 0.3,
                "candidates": [],
            }},
            image_quality={"blur": True, "glare": True, "tilt_deg": 30, "not_legible": False},
        )
        kwh_r = next(r for r in result["results"] if r["field"] == "kWh")
        assert len(kwh_r["reason"]) <= 120
