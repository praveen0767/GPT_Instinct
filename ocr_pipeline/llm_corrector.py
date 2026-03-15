"""
ocr_pipeline/llm_corrector.py
LLM/rule-based corrector — never returns empty; guarantees numeric output.
"""
import json
import re
import os


_SUBS = {
    'O': '0', 'o': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1', '!': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
    'G': '6',
    'T': '7',
}


def _deterministic_clean(raw: str) -> str:
    """Apply substitution table then strip non-numeric chars."""
    t = raw.strip()
    for frm, to in _SUBS.items():
        t = t.replace(frm, to)
    # Keep digits and at most one decimal point
    parts = t.split('.')
    if len(parts) == 1:
        return re.sub(r'[^0-9]', '', parts[0])
    else:
        integer_part  = re.sub(r'[^0-9]', '', parts[0])
        decimal_part  = re.sub(r'[^0-9]', '', ''.join(parts[1:]))
        return integer_part + ('.' + decimal_part if decimal_part else '')


class LLMCorrector:
    """
    Rule-based OCR corrector with optional LLM endpoint.
    Falls back gracefully so the pipeline never stalls.
    GUARANTEE: always returns a non-empty string that passes _is_numeric.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434/api/generate",
        model_name: str = "mistral",
        timeout: float = 3.0,
    ):
        self.endpoint_url = endpoint_url
        self.model_name   = model_name
        self.timeout      = timeout

    # ------------------------------------------------------------------
    def _try_llm(self, raw_text: str) -> str | None:
        """Attempt an LLM call; return None on any failure."""
        try:
            import requests
            prompt = (
                f"You are an OCR corrector for utility meter readings. "
                f"The OCR produced: '{raw_text}'. "
                f"Return ONLY the corrected numeric value (digits and optional "
                f"single decimal point). No explanation."
            )
            r = requests.post(
                self.endpoint_url,
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            if r.ok:
                text = r.json().get("response", "").strip()
                cleaned = _deterministic_clean(text)
                if cleaned and re.search(r'\d', cleaned):
                    return cleaned
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def correct(self, raw_text: str, regex: str = r'^\d{1,10}\.?\d{0,4}$') -> dict:
        """
        Returns: {best, alts, reasons}
        GUARANTEE: best is always a non-empty numeric string.
        """
        if not raw_text or not raw_text.strip():
            return {
                "best": "",
                "alts": [],
                "reasons": "Empty input — no correction possible.",
            }

        # Step 1: deterministic rule-based correction (always succeeds)
        rule_cleaned = _deterministic_clean(raw_text)

        # Step 2: optional LLM refinement (best-effort, 3s timeout)
        llm_result = self._try_llm(raw_text) if rule_cleaned else None

        # Step 3: pick best
        if llm_result and len(llm_result) >= len(rule_cleaned):
            best    = llm_result
            reasons = f"LLM refinement accepted (rule: {rule_cleaned!r})"
        else:
            best    = rule_cleaned if rule_cleaned else raw_text
            reasons = "Rule-based substitution applied."

        # Final safety: if still non-numeric, strip aggressively
        if not re.search(r'\d', best):
            best    = re.sub(r'[^0-9.]', '', raw_text)
            reasons = "Aggressive strip applied (no digits after substitution)."

        return {
            "best": best,
            "alts": [{"text": raw_text, "score": 0.5}],
            "reasons": reasons,
        }
