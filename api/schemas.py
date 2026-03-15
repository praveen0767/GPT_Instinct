from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class CandidateDetail(BaseModel):
    value: str
    score: float

class FieldOutput(BaseModel):
    value: str
    probability: float
    sources: Optional[List[str]] = None
    decimals: Optional[int] = None
    candidates: Optional[List[CandidateDetail]] = None
    debug: Optional[Dict[str, Any]] = None
    # Post-processor enrichment (v2.1)
    decimal_confidence: Optional[float] = None
    decimal_position: Optional[int] = None
    flags: Optional[List[str]] = None
    reason: Optional[str] = None

class ImageQualityFlags(BaseModel):
    blur: bool
    glare: bool
    tilt_deg: float
    not_legible: bool

class ArtifactURIs(BaseModel):
    crop_url: Optional[str] = None
    color_mask_url: Optional[str] = None
    alignment_map: Optional[str] = None
    glare_mask: Optional[str] = None
    sr_url: Optional[str] = None
    model_outputs: Optional[str] = None

class OCRResponseSchema(BaseModel):
    image_id: str
    meter_serial: FieldOutput
    kwh: FieldOutput
    kvah: FieldOutput
    md_kw: FieldOutput
    demand_kva: FieldOutput
    image_quality: ImageQualityFlags
    reason_codes: Optional[List[str]] = []
    qc_flag: bool
    processing_latency_ms: int
    artifacts: ArtifactURIs
