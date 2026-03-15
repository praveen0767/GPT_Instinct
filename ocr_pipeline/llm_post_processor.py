"""
ocr_pipeline/llm_post_processor.py
Structured OCR post-processor — deterministic calibration + optional LLM.

Replaces the loose chain of LLMCorrector → DecimalValidator → ModelCalibrator
with a single orchestrated step that produces calibrated confidence, decimal
confidence, domain-validated outputs, flag codes, and QC gating.
"""
import json
import math
import re
import os
from typing import Dict, List, Optional, Any

from ocr_pipeline.post_processor_schemas import (
    PostProcessorInput, PostProcessorOutput, FieldResult,
    RawOCRField, OCRCandidate, ImageQualityInput,
    DomainRule, EnsembleInfo, HistoricalCalibration,
)

# ── Tunable hyperparameters ──────────────────────────────────────────────────

LOGISTIC_STEEPNESS      = 6.0       # sigmoid steepness for calibration
LOGISTIC_CENTER         = 0.5       # sigmoid center
IQ_WEIGHT               = 0.3       # image-quality factor weight
CANDIDATE_WEIGHT        = 0.7       # OCR candidate score weight
DECIMAL_AMBIG_MULT      = 0.85      # decimal-confidence reduction on ambiguity
DOMAIN_VIOLATION_MULT   = 0.5       # confidence reduction on range violation
SEVERE_DOMAIN_MULT      = 0.4       # confidence reduction on severe violation
QC_THRESHOLD            = 0.80      # overall_pass confidence threshold
CHAR_CONF_PENALTY       = 0.02      # per-character correction penalty

# ── Character substitution table (OCR confusion) ────────────────────────────

_SUBS = {
    'O': '0', 'o': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1', '!': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
    'G': '6',
    'T': '7',
}

# ── Default domain rules ────────────────────────────────────────────────────

DEFAULT_DOMAIN_RULES: Dict[str, dict] = {
    "kWh":        {"unit": "kWh",  "min": 0, "max": 999999, "decimals_allowed": 1},
    "kVAh":       {"unit": "kVAh", "min": 0, "max": 999999, "decimals_allowed": 1},
    "MD_kW":      {"unit": "kW",   "min": 0, "max": 9999,   "decimals_allowed": 2},
    "Demand_kVA": {"unit": "kVA",  "min": 0, "max": 9999,   "decimals_allowed": 2},
    "serial":     {"pattern": r"^[A-Z0-9\-]{4,20}$"},
}

# Field name normalization: pipeline uses lowercase, spec uses mixed case
_FIELD_MAP = {
    "kwh": "kWh", "kvah": "kVAh", "md_kw": "MD_kW",
    "demand_kva": "Demand_kVA", "meter_serial": "serial",
}
_FIELD_MAP_REV = {v: k for k, v in _FIELD_MAP.items()}

_HAS_DIGIT = re.compile(r'\d')


# ═══════════════════════════════════════════════════════════════════════════════
# OCRPostProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class OCRPostProcessor:
    """
    Structured OCR post-processor implementing the full specification:
    - Per-field calibrated confidence (logistic squash)
    - Decimal-placement confidence with ambiguity detection
    - Domain validation (range, pattern, decimal count)
    - Flag codes (UNCERT_DECIMAL, LOW_CONFIDENCE, etc.)
    - QC gating (overall_pass)
    - Optional LLM refinement (Ollama/vLLM, 3s timeout)
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434/api/generate",
        model_name: str = "mistral",
        timeout: float = 3.0,
        domain_rules: Optional[Dict[str, dict]] = None,
    ):
        self.endpoint_url = endpoint_url
        self.model_name   = model_name
        self.timeout      = timeout
        self.domain_rules = domain_rules or DEFAULT_DOMAIN_RULES

        # Load system prompt if available
        self._system_prompt = self._load_system_prompt()

    # ── Public API ────────────────────────────────────────────────────────

    def process(
        self,
        image_id: str,
        raw_ocr_fields: Dict[str, dict],
        image_quality: Optional[dict] = None,
        preproc_info: Optional[dict] = None,
        ensemble_info: Optional[dict] = None,
    ) -> dict:
        """
        Main entry point. Takes OCR results from multi_field_ocr and produces
        the full structured output.

        Parameters
        ----------
        image_id : str
        raw_ocr_fields : dict mapping field_name → {value, probability, candidates, debug, ...}
        image_quality : dict from analyze_image_quality() {blur, glare, tilt_deg, not_legible}
        preproc_info : dict with dewarping/contrast info
        ensemble_info : optional model vote weights

        Returns
        -------
        dict matching PostProcessorOutput schema
        """
        iq = self._normalize_image_quality(image_quality or {})
        results: List[dict] = []
        qc_reasons: List[str] = []

        # Check image-level readability
        if not iq.get("readable_to_human", True):
            for field_name in raw_ocr_fields:
                spec_field = _FIELD_MAP.get(field_name, field_name)
                results.append(FieldResult(
                    field=spec_field,
                    raw_best=None,
                    corrected=None,
                    normalized_value=None,
                    unit=self._get_unit(spec_field),
                    confidence=0.0,
                    decimal_confidence=None,
                    decimal_position=None,
                    flags=["IMAGE_QUALITY_POOR", "HUMAN_REVIEW_REQUIRED"],
                    reason="Image not readable to human — skipped.",
                ).model_dump())
            qc_reasons.append("IMAGE_QUALITY_POOR")
            return PostProcessorOutput(
                image_id=image_id,
                results=[FieldResult(**r) for r in results],
                overall_pass=False,
                qc_reasons=qc_reasons,
            ).model_dump()

        # Process each field
        for field_name, field_data in raw_ocr_fields.items():
            spec_field = _FIELD_MAP.get(field_name, field_name)
            result = self._process_field(spec_field, field_data, iq, ensemble_info)
            results.append(result)

        # Determine overall_pass
        all_confident = True
        for r in results:
            conf = r.get("confidence") or 0.0
            if conf < QC_THRESHOLD:
                all_confident = False
            flags = r.get("flags", [])
            if "HUMAN_REVIEW_REQUIRED" in flags:
                if "HUMAN_REVIEW_REQUIRED" not in qc_reasons:
                    qc_reasons.append("HUMAN_REVIEW_REQUIRED")
            if "LOW_CONFIDENCE" in flags:
                if "LOW_CONFIDENCE" not in qc_reasons:
                    qc_reasons.append("LOW_CONFIDENCE")
            if "UNCERT_DECIMAL" in flags:
                if "UNCERT_DECIMAL" not in qc_reasons:
                    qc_reasons.append("UNCERT_DECIMAL")
            if "OUT_OF_RANGE" in flags:
                if "OUT_OF_RANGE" not in qc_reasons:
                    qc_reasons.append("OUT_OF_RANGE")
            if "IMAGE_QUALITY_POOR" in flags:
                if "IMAGE_QUALITY_POOR" not in qc_reasons:
                    qc_reasons.append("IMAGE_QUALITY_POOR")

        overall_pass = all_confident and len(qc_reasons) == 0

        return PostProcessorOutput(
            image_id=image_id,
            results=[FieldResult(**r) for r in results],
            overall_pass=overall_pass,
            qc_reasons=qc_reasons,
        ).model_dump()

    # ── Per-field processing ──────────────────────────────────────────────

    def _process_field(
        self,
        spec_field: str,
        field_data: dict,
        iq: dict,
        ensemble_info: Optional[dict],
    ) -> dict:
        """Process one field and return a FieldResult-compatible dict."""
        value = field_data.get("value", "")
        probability = field_data.get("probability", 0.0)
        candidates_raw = field_data.get("candidates", [])
        debug = field_data.get("debug", {})
        flags: List[str] = []
        reason_parts: List[str] = []

        # Empty / no-data case
        if not value or value in ("—", "N/A", ""):
            return {
                "field": spec_field,
                "raw_best": None,
                "corrected": None,
                "normalized_value": None,
                "unit": self._get_unit(spec_field),
                "confidence": 0.0,
                "decimal_confidence": None,
                "decimal_position": None,
                "flags": ["MISSING_DIGITS"],
                "reason": "No numeric data found for this field.",
                "debug_notes": None,
            }

        raw_best = value
        is_serial = (spec_field == "serial")

        # ── Character-level correction ────────────────────────────────
        if is_serial:
            corrected = value  # don't apply digit substitutions to serial
        else:
            corrected, corrections = self._apply_char_corrections(value)
            if corrections:
                reason_parts.append("; ".join(corrections[:3]))

        # ── Candidates analysis ───────────────────────────────────────
        candidate_texts = [c.get("value", c.get("text", "")) for c in candidates_raw]
        candidate_confs = [c.get("score", c.get("conf", 0.0)) for c in candidates_raw]

        # ── Calibrated confidence ─────────────────────────────────────
        iq_factor = self._compute_iq_factor(iq)
        candidate_score = probability  # from ROVER ensemble

        # Check for historical calibration
        hist_cal = None
        if ensemble_info and ensemble_info.get("historical_calibration"):
            for hc in ensemble_info["historical_calibration"]:
                if hc.get("field") == spec_field:
                    hist_cal = hc
                    break

        calibrated = self._calibrate(candidate_score, iq_factor, hist_cal)

        # ── Domain validation ─────────────────────────────────────────
        rule = self._get_domain_rule(spec_field)

        if is_serial:
            # Pattern validation
            pattern = rule.get("pattern") if rule else None
            if pattern and not re.match(pattern, corrected):
                flags.append("NONMATCH_SERIAL")
                reason_parts.append("Serial doesn't match expected pattern")
                calibrated *= DOMAIN_VIOLATION_MULT

            return {
                "field": spec_field,
                "raw_best": raw_best,
                "corrected": corrected,
                "normalized_value": None,
                "unit": None,
                "confidence": round(calibrated, 4),
                "decimal_confidence": None,
                "decimal_position": None,
                "flags": flags,
                "reason": self._build_reason(reason_parts, "pattern match + high confidence"),
                "debug_notes": None,
            }

        # ── Numeric field processing ──────────────────────────────────

        # Parse numeric value
        normalized = self._parse_numeric(corrected)

        # Range validation
        if rule and normalized is not None:
            vmin = rule.get("min")
            vmax = rule.get("max")
            if vmin is not None and normalized < vmin:
                flags.append("OUT_OF_RANGE")
                reason_parts.append(f"Below min {vmin}")
                calibrated *= SEVERE_DOMAIN_MULT
            if vmax is not None and normalized > vmax:
                flags.append("OUT_OF_RANGE")
                reason_parts.append(f"Above max {vmax}")
                calibrated *= SEVERE_DOMAIN_MULT

        # ── Decimal analysis ──────────────────────────────────────────
        decimal_pos = self._get_decimal_position(corrected)
        decimal_conf = calibrated  # start at calibrated confidence

        # Check decimal ambiguity among candidates
        decimal_positions = set()
        for ct in candidate_texts:
            dp = self._get_decimal_position(ct)
            if dp is not None:
                decimal_positions.add(dp)
        # Also consider the current corrected value
        if decimal_pos is not None:
            decimal_positions.add(decimal_pos)

        if len(decimal_positions) > 1:
            # Ambiguous decimal across candidates
            decimal_conf = min(0.98, calibrated * DECIMAL_AMBIG_MULT)
            flags.append("UNCERT_DECIMAL")
            reason_parts.append("Conflicting decimal positions among candidates")
            if calibrated < 0.8:
                flags.append("HUMAN_REVIEW_REQUIRED")

        # Check decimals_allowed
        if rule and decimal_pos is not None:
            allowed = rule.get("decimals_allowed")
            if allowed is not None and decimal_pos > allowed:
                flags.append("UNCERT_DECIMAL")
                decimal_conf *= 0.8
                reason_parts.append(f"Decimal places {decimal_pos} > allowed {allowed}")

        # Multi-dot detection
        if corrected.count('.') > 1:
            flags.append("MULTI_DOT")
            decimal_conf = min(decimal_conf, 0.3)
            reason_parts.append("Multiple decimal points detected")

        # ── Confidence gating ─────────────────────────────────────────
        if calibrated < QC_THRESHOLD:
            if "LOW_CONFIDENCE" not in flags:
                flags.append("LOW_CONFIDENCE")

        # Image quality degradation
        if iq.get("brightness", 0.5) < 0.2 or iq.get("sharpness", 0.5) < 0.3:
            if "IMAGE_QUALITY_POOR" not in flags:
                flags.append("IMAGE_QUALITY_POOR")

        # ── Conflicting candidates ────────────────────────────────────
        unique_texts = set(ct for ct in candidate_texts if ct)
        if len(unique_texts) > 2:
            if "CONFLICTING_CANDIDATES" not in flags:
                flags.append("CONFLICTING_CANDIDATES")

        # Default reason
        default_reason = (
            f"ensemble & IQ agree" if calibrated >= 0.9
            else f"calibrated conf={calibrated:.2f}"
        )

        return {
            "field": spec_field,
            "raw_best": raw_best,
            "corrected": corrected,
            "normalized_value": normalized,
            "unit": self._get_unit(spec_field),
            "confidence": round(min(1.0, max(0.0, calibrated)), 4),
            "decimal_confidence": round(min(1.0, max(0.0, decimal_conf)), 4) if decimal_conf is not None else None,
            "decimal_position": decimal_pos,
            "flags": flags,
            "reason": self._build_reason(reason_parts, default_reason),
            "debug_notes": None,
        }

    # ── Calibration ───────────────────────────────────────────────────────

    def _calibrate(
        self,
        candidate_score: float,
        iq_factor: float,
        historical: Optional[dict] = None,
    ) -> float:
        """
        Compute calibrated probability using the spec's logistic squash.

        raw_prob = candidate_score * 0.7 + iq_factor * 0.3
        calibrated = 1 / (1 + exp(-steepness * (raw_prob - center)))
        """
        raw_prob = candidate_score * CANDIDATE_WEIGHT + iq_factor * IQ_WEIGHT

        if historical:
            # Use historical logistic parameters
            a = historical.get("logistic_a", -4.0)
            b = historical.get("logistic_b", 5.2)
            return 1.0 / (1.0 + math.exp(-(a + b * raw_prob)))

        # Default logistic squash
        return 1.0 / (1.0 + math.exp(-LOGISTIC_STEEPNESS * (raw_prob - LOGISTIC_CENTER)))

    def _compute_iq_factor(self, iq: dict) -> float:
        """image_quality_factor = mean(brightness, sharpness) * (1 - glare)"""
        brightness = iq.get("brightness", 0.5)
        sharpness  = iq.get("sharpness", 0.5)
        glare      = iq.get("glare", 0.0)
        return ((brightness + sharpness) / 2.0) * (1.0 - glare)

    # ── Character correction ──────────────────────────────────────────────

    def _apply_char_corrections(self, text: str) -> tuple:
        """
        Apply substitution table. Returns (corrected_text, [corrections]).
        Each correction is a string like "O→0 at pos 3".
        """
        result = list(text)
        corrections = []
        for i, ch in enumerate(result):
            if ch in _SUBS:
                replacement = _SUBS[ch]
                corrections.append(f"{ch}→{replacement} at pos {i}")
                result[i] = replacement
        corrected = ''.join(result)

        # Strip non-numeric (keep digits and single decimal)
        parts = corrected.split('.')
        if len(parts) == 1:
            corrected = re.sub(r'[^0-9]', '', parts[0])
        elif len(parts) == 2:
            integer_part = re.sub(r'[^0-9]', '', parts[0])
            decimal_part = re.sub(r'[^0-9]', '', parts[1])
            corrected = integer_part + ('.' + decimal_part if decimal_part else '')
        else:
            # Multiple dots — keep first two segments
            integer_part = re.sub(r'[^0-9]', '', parts[0])
            decimal_part = re.sub(r'[^0-9]', '', ''.join(parts[1:]))
            corrected = integer_part + ('.' + decimal_part if decimal_part else '')

        return corrected, corrections

    # ── Domain helpers ────────────────────────────────────────────────────

    def _get_domain_rule(self, spec_field: str) -> Optional[dict]:
        return self.domain_rules.get(spec_field)

    def _get_unit(self, spec_field: str) -> Optional[str]:
        rule = self._get_domain_rule(spec_field)
        if rule:
            return rule.get("unit")
        return None

    def _get_decimal_position(self, text: str) -> Optional[int]:
        """
        Returns decimal position as count of digits after the decimal point.
        Returns 0 if no decimal (integer). Returns None if text is empty.
        """
        if not text:
            return None
        if '.' not in text:
            return 0
        parts = text.split('.')
        if len(parts) != 2:
            return None
        return len(parts[1])

    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse cleaned text to float. Returns None on failure."""
        if not text:
            return None
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    # ── Image quality normalization ───────────────────────────────────────

    def _normalize_image_quality(self, iq: dict) -> dict:
        """
        Convert pipeline image_quality dict (blur/glare/tilt_deg/not_legible)
        to the spec format (brightness/glare/sharpness/tilt_deg/readable_to_human).
        """
        blur_flag = iq.get("blur", False)
        glare_flag = iq.get("glare", False)
        tilt_deg = iq.get("tilt_deg", 0.0)
        not_legible = iq.get("not_legible", False)

        return {
            "brightness": 0.3 if blur_flag else 0.7,
            "glare": 0.7 if glare_flag else 0.1,
            "sharpness": 0.2 if blur_flag else 0.8,
            "tilt_deg": float(tilt_deg),
            "readable_to_human": not not_legible,
        }

    # ── Reason builder ────────────────────────────────────────────────────

    def _build_reason(self, parts: List[str], default: str) -> str:
        """Build reason string, capped at 120 chars."""
        if parts:
            combined = "; ".join(parts)
            return combined[:120]
        return default[:120]

    # ── Optional LLM call ─────────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file, or return a compact fallback."""
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "post_processor_system.txt"
        )
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, OSError):
            return (
                "You are an OCR post-processor for utility meter readings. "
                "Return ONLY valid JSON matching the required schema. "
                "temperature=0, deterministic outputs only."
            )

    def try_llm_refinement(
        self,
        input_payload: dict,
    ) -> Optional[dict]:
        """
        Attempt LLM call with the full spec prompt. Returns parsed output or None.
        This is optional — the deterministic path always works.
        """
        try:
            import requests
            prompt = (
                f"Process this OCR data and return the corrected JSON output:\n\n"
                f"{json.dumps(input_payload, indent=2)}"
            )
            r = requests.post(
                self.endpoint_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": self._system_prompt,
                    "stream": False,
                    "options": {"temperature": 0, "top_p": 1.0},
                },
                timeout=self.timeout,
            )
            if r.ok:
                text = r.json().get("response", "").strip()
                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if "results" in parsed:
                        return parsed
        except Exception:
            pass
        return None
