"""
meter_ocr/validators/domain_rules.py
Energy meter domain rule validator.

Accepted ranges (per spec):
  kWh:        0 – 9,999,999
  kVAh:       0 – 9,999,999
  MD kW:      0 – 999.9
  Demand kVA: 0 – 999.9
  meter_serial: any non-empty numeric string (no range restriction)

Any value outside range → reject (return empty) and add reason_code.
"""
from typing import Optional, Tuple

FIELD_RANGES = {
    "kwh":        (0.0, 9_999_999.0),
    "kvah":       (0.0, 9_999_999.0),
    "md_kw":      (0.0,       999.9),
    "demand_kva": (0.0,       999.9),
    "meter_serial": None,              # no numeric range restriction
}


def validate_field(field: str, value: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_valid, reason_code_or_None).

    reason_code examples: "OUT_OF_RANGE_KWH", "EMPTY_VALUE", "NON_NUMERIC"
    """
    if not value or value.strip() in ("", "—", "N/A"):
        return False, f"EMPTY_VALUE_{field.upper()}"

    # Numeric check
    try:
        num = float(value.replace(",", ""))
    except ValueError:
        return False, f"NON_NUMERIC_{field.upper()}"

    # Range check
    rng = FIELD_RANGES.get(field)
    if rng is not None:
        lo, hi = rng
        if not (lo <= num <= hi):
            return False, f"OUT_OF_RANGE_{field.upper()}"

    return True, None


def apply_domain_rules(raw_values: dict) -> dict:
    """
    Parameters
    ----------
    raw_values : { field: value_string, ... }

    Returns
    -------
    {
      "validated": { field: value_or_empty },
      "reason_codes": [str, ...],
    }
    """
    validated    = {}
    reason_codes = []

    for field in ("kwh", "kvah", "md_kw", "demand_kva", "meter_serial"):
        val = str(raw_values.get(field, "")).strip()
        ok, code = validate_field(field, val)
        if ok:
            validated[field] = val
        else:
            validated[field] = "—"
            if code:
                reason_codes.append(code)

    return {"validated": validated, "reason_codes": reason_codes}
