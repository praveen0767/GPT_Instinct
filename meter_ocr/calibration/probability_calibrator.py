"""
meter_ocr/calibration/probability_calibrator.py
Isotonic regression confidence calibrator.

Inputs  : raw_ocr_conf, engine_agreement, decimal_conf, image_quality_score
Outputs : calibrated probability ∈ [0, 1]

Targets (per spec):
  median  ≥ 0.995
  p5      ≥ 0.980
"""
import os
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression


def _feature_vector(
    raw_conf: float,
    engine_agreement: float,
    decimal_conf: float,
    image_quality_score: float,
) -> float:
    """
    Combine 4 inputs into a single scalar score for isotonic calibration.
    Weighted average where agreement is most important.
    """
    return (
        0.40 * raw_conf +
        0.30 * engine_agreement +
        0.20 * decimal_conf +
        0.10 * image_quality_score
    )


def _synthetic_data(n: int = 1000):
    """
    Synthetic (combined_score, label) pairs that model realistic OCR accuracy:
    - High combined score → usually correct
    - Low combined score  → often wrong
    """
    rng     = np.random.default_rng(42)
    scores  = rng.beta(6, 1.5, n)
    p_corr  = 1 / (1 + np.exp(-10 * (scores - 0.45)))
    labels  = rng.binomial(1, p_corr).astype(float)
    return scores, labels


class ProbabilityCalibrator:
    """
    Per-field isotonic regression calibrator.

    Usage
    -----
        cal = ProbabilityCalibrator(field="kwh")
        p   = cal.calibrate(raw_conf=0.82, engine_agreement=0.67,
                            decimal_conf=0.90, image_quality_score=0.85)
    """

    def __init__(self, field: str = "default", save_dir: str = "data/calibration"):
        self.field     = field
        self.save_dir  = save_dir
        self._path     = os.path.join(save_dir, f"{field}_isotonic.pkl")
        self._reg      = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.fitted    = False

        # Load saved calibrator if present
        if os.path.isfile(self._path):
            try:
                with open(self._path, "rb") as f:
                    self._reg = pickle.load(f)
                self.fitted = True
                print(f"Calibrator[{field}]: loaded from {self._path}")
            except Exception as e:
                print(f"Calibrator[{field}]: load error ({e}), fitting synthetic.")

        if not self.fitted:
            self._fit_synthetic()

    # ------------------------------------------------------------------
    def _fit_synthetic(self):
        scores, labels = _synthetic_data(n=1200)
        self._reg.fit(scores, labels)
        self.fitted = True
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(self._path, "wb") as f:
                pickle.dump(self._reg, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def fit(self, raw_confs, engine_agreements, decimal_confs,
            iq_scores, labels):
        """Fit on real labelled data (call once you have ground truth)."""
        Xs = np.array([
            _feature_vector(r, a, d, q)
            for r, a, d, q in zip(raw_confs, engine_agreements, decimal_confs, iq_scores)
        ])
        self._reg.fit(Xs, np.asarray(labels, dtype=float))
        self.fitted = True
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._reg, f)

    # ------------------------------------------------------------------
    def calibrate(
        self,
        raw_conf:            float,
        engine_agreement:    float = 0.67,
        decimal_conf:        float = 0.80,
        image_quality_score: float = 0.90,
    ) -> float:
        """Return calibrated probability ∈ [0, 1]."""
        x    = _feature_vector(raw_conf, engine_agreement, decimal_conf, image_quality_score)
        prob = float(self._reg.transform([x])[0])
        return round(min(1.0, max(0.0, prob)), 6)


# ── Convenience: per-field calibrators ────────────────────────────────────────

class FieldCalibrators:
    FIELDS = ["kwh", "kvah", "md_kw", "demand_kva", "meter_serial"]

    def __init__(self, save_dir: str = "data/calibration"):
        self._cals = {f: ProbabilityCalibrator(field=f, save_dir=save_dir)
                      for f in self.FIELDS}

    def calibrate(self, field: str, **kwargs) -> float:
        c = self._cals.get(field, self._cals["kwh"])
        return c.calibrate(**kwargs)
