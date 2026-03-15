"""
ocr_pipeline/trocr_adapter.py
TrOCR adapter — strict numeric output, proper CPU inference, graceful fallback.
"""
import re
import numpy as np

_HAS_DIGIT = re.compile(r'\d')

def _clean(text: str) -> str:
    t = text.strip()
    for frm, to in [("O","0"),("o","0"),("I","1"),("l","1"),("S","5"),("B","8"),("Z","2"),("z","2")]:
        t = t.replace(frm, to)
    t = re.sub(r'[^0-9.]', '', t)
    return t

def _is_valid(text: str) -> bool:
    if not _HAS_DIGIT.search(text):
        return False
    parts = text.split(".")
    if len(parts) > 2:
        return False
    return all(p.isdigit() for p in parts if p)


class TrOCRAdapter:
    """Adapter for Microsoft TrOCR with strict numeric output."""

    # Use the printed model — significantly better on digits than stage1
    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.device = device
        self.mock_mode = False
        model_name = model_name or self.DEFAULT_MODEL

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)

            # Force any meta-device tensors to CPU (workaround for some HF versions)
            for name, param in self.model.named_parameters():
                if param.device.type == 'meta':
                    print(f"  TrOCR: forcing {name} from meta → cpu")
                    param.data = param.data.to('cpu')

            self.model.eval()
            self._torch = torch
            print(f"TrOCRAdapter loaded: {model_name} on {self.device}")

        except Exception as e:
            print(f"Warning: TrOCR load failed ({e}). Adapter disabled.")
            self.mock_mode = True

    # ------------------------------------------------------------------
    def recognize(self, image: np.ndarray) -> dict:
        """
        Run TrOCR inference and return numeric reading only.
        On failure returns empty result (never raises).
        """
        if self.mock_mode:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

        try:
            from PIL import Image
            pil_image = Image.fromarray(image).convert("RGB")
            pixel_values = self.processor(
                pil_image, return_tensors="pt"
            ).pixel_values.to(self.device)

            with self._torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=20,
                    use_cache=False,
                    num_beams=4,           # beam search for better accuracy
                    early_stopping=True,
                )

            generated_ids  = outputs.sequences
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Token-level confidence
            scores      = outputs.scores
            token_probs = [
                self._torch.softmax(s, dim=-1).max().item() for s in scores
            ]
            avg_conf = float(sum(token_probs) / len(token_probs)) if token_probs else 0.0

            cleaned = _clean(generated_text)

            if not cleaned or not _is_valid(cleaned):
                # TrOCR returned non-numeric — return empty so ROVER ignores it
                return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}

            return {
                "text": cleaned,
                "confidence": avg_conf,
                "tokens": list(cleaned),
                "token_scores": token_probs,
            }

        except Exception as e:
            print(f"TrOCR inference error: {e}")
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}
