"""
ocr_pipeline/serial_ocr.py
Step 11: Serial OCR (Alphanumeric focus)

Takes the YOLOv8 identified `nameplate_serial` bounding box and runs
alnum-focused OCR exclusively on it to combat the missing serial extraction error.
"""
import re
from typing import Dict
from ocr_pipeline.paddle_adapter import PaddleAdapter
from ocr_pipeline.trocr_adapter import TrOCRAdapter

class SerialOCREngine:
    def __init__(self):
        self.paddle = PaddleAdapter()
        self.trocr = TrOCRAdapter()
        
        # Master Prompt V3 strict character confusion guidelines
        self.confusion_map = {"O":"0","I":"1","L":"1","S":"5","B":"8","Z":"2"}

    def _apply_confusion_recovery(self, s: str) -> str:
        s = s.upper()
        s = re.sub(r"[^A-Z0-9\-]", "", s)
        # Apply corrections heavily if length is suspiciously short
        if len(s) < 4:
            s2 = "".join(self.confusion_map.get(ch,ch) for ch in s)
            if len(s2) >= 4: 
                return s2
        return s
        
    def extract_serial(self, nameplate_crop) -> Dict:
        """Run OCR focused on alphanumeric serial numbers on isolated crop."""
        if nameplate_crop is None or nameplate_crop.size == 0:
            return {"text": "", "confidence": 0.0}
            
        p_res = self.paddle.recognize(nameplate_crop)
        t_res = self.trocr.recognize(nameplate_crop)
        
        # Simple confidence voting for Alphanumeric text
        best_res = p_res if p_res.get("confidence", 0) > t_res.get("confidence", 0) else t_res
        raw_text = best_res.get("text", "")
        
        # Apply character heuristic correction targeted for Serials
        fixed_text = self._apply_confusion_recovery(raw_text)
        
        return {
            "text": fixed_text,
            "confidence": best_res.get("confidence", 0.0),
            "sources": ["paddleocr", "trocr"]
        }
