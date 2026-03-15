"""
meter_ocr/ocr/trocr_reader.py
TrOCR OCR reader — numeric display extraction only.
Model: microsoft/trocr-base-printed  (best for printed digits)
Weight in ensemble: 0.5 (primary)
"""
import re
import numpy as np

_SUBS = {'O':'0','o':'0','I':'1','l':'1','S':'5','B':'8','Z':'2','z':'2','G':'6'}
_DIGIT_RE = re.compile(r'\d')


def _clean(text: str) -> str:
    t = text.strip()
    for f, r in _SUBS.items():
        t = t.replace(f, r)
    # Keep only digits and at most one decimal
    parts = t.split('.')
    if len(parts) == 1:
        return re.sub(r'[^0-9]', '', parts[0])
    return re.sub(r'[^0-9]', '', parts[0]) + '.' + re.sub(r'[^0-9]', '', ''.join(parts[1:]))


class TrOCRReader:
    """
    Applies TrOCR to a pre-cropped display image.
    Always returns only numeric text.
    """
    MODEL_ID = "microsoft/trocr-base-printed"

    def __init__(self, model_id: str = None, device: str = "cpu"):
        model_id    = model_id or self.MODEL_ID
        self.device = device
        self._ok    = False
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            self.processor = TrOCRProcessor.from_pretrained(model_id)
            self.model     = VisionEncoderDecoderModel.from_pretrained(model_id)
            self.model.to(device)
            self.model.eval()
            self._torch = torch
            self._ok = True
            print(f"TrOCRReader ready: {model_id} on {device}")
        except Exception as e:
            print(f"TrOCRReader load failed ({e}). Disabled.")

    # ------------------------------------------------------------------
    def read(self, image: np.ndarray) -> dict:
        """
        Parameters
        ----------
        image : BGR numpy array — ALREADY CROPPED display region.

        Returns
        -------
        {text, confidence, weight}
        """
        if not self._ok:
            return {"text": "", "confidence": 0.0, "weight": 0.5}
        try:
            from PIL import Image
            pil = Image.fromarray(image[:, :, ::-1]).convert("RGB")
            px  = self.processor(pil, return_tensors="pt").pixel_values.to(self.device)
            with self._torch.no_grad():
                out = self.model.generate(
                    px,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=4,
                    max_new_tokens=20,
                    early_stopping=True,
                )
            text    = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
            probs   = [self._torch.softmax(s, -1).max().item() for s in out.scores]
            avg_conf= float(sum(probs) / len(probs)) if probs else 0.0
            cleaned = _clean(text)
            if not cleaned or not _DIGIT_RE.search(cleaned):
                return {"text": "", "confidence": 0.0, "weight": 0.5}
            return {"text": cleaned, "confidence": avg_conf, "weight": 0.5}
        except Exception as e:
            print(f"TrOCRReader inference error: {e}")
            return {"text": "", "confidence": 0.0, "weight": 0.5}
