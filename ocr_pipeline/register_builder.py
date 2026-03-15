"""
ocr_pipeline/register_builder.py
Step 9 & 10: Register Reconstruction & Decimal Handling

Fuses sequential CNN predictions (digits) and the Decimal detector
into a numeric value string.
"""
from typing import List, Dict, Optional

class RegisterBuilder:
    def __init__(self):
        pass

    def build_register(
        self, 
        digit_preds: List[Dict], 
        decimal_position: Optional[int] = None
    ) -> Dict:
        """
        Input constraint: digit_preds is geographically sorted left-to-right.
        Example: [{'digit': '1', 'prob': 0.99}, {'digit': '4', 'prob': 0.92}]
        Output: "14" or "1.4" if decimal_position is provided.
        """
        if not digit_preds:
            return {
                "raw_text": "",
                "confidence": 0.0,
                "components": []
            }
            
        digits_str = "".join([dp["digit"] for dp in digit_preds])
        avg_conf = sum(dp["prob"] for dp in digit_preds) / len(digit_preds)
        
        # Insert decimal if positioned by external CNN (from right)
        # e.g., "12345" at pos 1 -> "1234.5"
        final_str = digits_str
        if decimal_position and 0 < decimal_position < len(digits_str):
            idx = len(digits_str) - decimal_position
            final_str = digits_str[:idx] + "." + digits_str[idx:]
            
        return {
            "raw_text": final_str,
            "confidence": avg_conf,
            "components": digit_preds
        }
