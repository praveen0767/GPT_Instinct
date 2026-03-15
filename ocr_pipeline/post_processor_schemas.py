"""
ocr_pipeline/post_processor_schemas.py
Pydantic models for the structured LLM post-processor I/O.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ── Input schemas ─────────────────────────────────────────────────────────────

class OCRCandidate(BaseModel):
    text: str
    conf: float = 0.0
    bbox: Optional[List[float]] = None


class RawOCRField(BaseModel):
    field: str
    candidates: List[OCRCandidate] = []


class ImageQualityInput(BaseModel):
    brightness: float = 0.5
    glare: float = 0.0
    sharpness: float = 0.5
    tilt_deg: float = 0.0
    readable_to_human: bool = True


class PreprocInfo(BaseModel):
    dewarped: bool = False
    contrast_normalized: bool = False
    crop_margin_px: int = 0


class DomainRule(BaseModel):
    unit: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    decimals_allowed: Optional[int] = None
    pattern: Optional[str] = None


class ModelVote(BaseModel):
    model: str
    weight: float = 1.0


class HistoricalCalibration(BaseModel):
    field: str
    logistic_a: float = -4.0
    logistic_b: float = 5.2


class EnsembleInfo(BaseModel):
    model_votes: List[ModelVote] = []
    historical_calibration: Optional[List[HistoricalCalibration]] = None


class PostProcessorInput(BaseModel):
    image_id: str
    raw_ocr: List[RawOCRField] = []
    image_quality: ImageQualityInput = ImageQualityInput()
    preproc: PreprocInfo = PreprocInfo()
    domain_rules: Dict[str, DomainRule] = {}
    ensemble_info: EnsembleInfo = EnsembleInfo()


# ── Output schemas ────────────────────────────────────────────────────────────

class FieldResult(BaseModel):
    field: str
    raw_best: Optional[str] = None
    corrected: Optional[str] = None
    normalized_value: Optional[float] = None
    unit: Optional[str] = None
    confidence: Optional[float] = None
    decimal_confidence: Optional[float] = None
    decimal_position: Optional[int] = None
    flags: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    debug_notes: Optional[str] = None


class PostProcessorOutput(BaseModel):
    image_id: str
    results: List[FieldResult] = []
    overall_pass: Optional[bool] = None
    qc_reasons: List[str] = Field(default_factory=list)
