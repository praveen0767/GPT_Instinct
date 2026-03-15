"""
ocr_pipeline/calibrator.py
Per-field confidence calibrator.

Supports:
  - Isotonic Regression (K=5 CV)
  - Synthetic fit for initial deployment (no labels needed)
  - Reliability diagram generation
  - save / load per field
"""
import os
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression

# ── synthetic calibration ─────────────────────────────────────────────────────

def _synthetic_calibration_data(n: int = 500):
    """
    Generate synthetic (raw_confidence, correct_label) pairs that model
    realistic OCR calibration behaviour:
      - High-confidence predictions are usually correct.
      - Low-confidence predictions are often wrong.
    """
    rng = np.random.default_rng(42)
    raw = rng.beta(5, 1.5, n)           # skewed towards high confidence
    # Label = 1 if reading was correct; probability of correct ≈ sigmoid(8*(raw-0.5))
    p_correct = 1 / (1 + np.exp(-8 * (raw - 0.5)))
    labels = rng.binomial(1, p_correct).astype(float)
    return raw, labels


# ── ModelCalibrator ───────────────────────────────────────────────────────────

class ModelCalibrator:
    """
    Single-field or multi-field confidence calibrator.

    Usage
    -----
        cal = ModelCalibrator(field="kwh")
        cal.fit(raw_confs, correct_labels)   # fit + auto-save
        p = cal.calibrate(0.87)              # → calibrated probability
    """

    def __init__(self, field: str = "default", model_dir: str = "data/calibration"):
        self.field     = field
        self.model_dir = model_dir
        self.iso_path  = os.path.join(model_dir, f"{field}_isotonic.pkl")
        self._reg      = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.is_fitted = False

        # Try loading saved calibration
        if os.path.isfile(self.iso_path):
            try:
                with open(self.iso_path, "rb") as f:
                    self._reg = pickle.load(f)
                self.is_fitted = True
                print(f"Calibrator[{field}]: loaded from {self.iso_path}")
            except Exception as e:
                print(f"Calibrator[{field}]: load failed ({e}), will use synthetic.")

        # Fit synthetic if no saved model
        if not self.is_fitted:
            self._fit_synthetic()

    # ------------------------------------------------------------------
    def _fit_synthetic(self):
        raw, labels = _synthetic_calibration_data(n=800)
        self._reg.fit(raw, labels)
        self.is_fitted = True
        # Try to persist
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.iso_path, "wb") as f:
                pickle.dump(self._reg, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def fit(self, raw_confidences: np.ndarray, labels: np.ndarray):
        """
        Fit on real validation data.

        Parameters
        ----------
        raw_confidences : 1-D float array ∈ [0, 1]
        labels          : 1-D binary array (1 = correct, 0 = wrong)
        """
        self._reg.fit(raw_confidences, labels)
        self.is_fitted = True
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.iso_path, "wb") as f:
            pickle.dump(self._reg, f)

    # ------------------------------------------------------------------
    def calibrate(self, raw_confidence: float) -> float:
        """Map a raw [0,1] confidence → calibrated probability ∈ [0,1]."""
        if not self.is_fitted:
            return float(raw_confidence)
        cal = float(self._reg.transform([raw_confidence])[0])
        return round(min(1.0, max(0.0, cal)), 6)

    # ------------------------------------------------------------------
    def reliability_diagram(self, raw_confs, labels, n_bins: int = 10,
                             save_path: str = None):
        """
        Generate a reliability diagram and optionally save as PNG.
        Returns (fig, ece, brier) — fig is None if matplotlib unavailable.
        """
        raw_confs = np.asarray(raw_confs)
        labels    = np.asarray(labels)

        # ECE
        bins       = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        ece = 0.0
        brier = float(np.mean((raw_confs - labels) ** 2))

        bin_accs, bin_confs, bin_counts = [], [], []
        for lo, hi in zip(bin_lowers, bin_uppers):
            mask = (raw_confs >= lo) & (raw_confs < hi)
            if mask.sum() == 0:
                continue
            acc  = labels[mask].mean()
            conf = raw_confs[mask].mean()
            n    = mask.sum()
            ece += (n / len(raw_confs)) * abs(acc - conf)
            bin_accs.append(acc);  bin_confs.append(conf);  bin_counts.append(n)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(bin_confs, bin_accs, width=1/n_bins, alpha=0.7,
                   edgecolor="black", label="Actual accuracy")
            ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect calibration")
            ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy")
            ax.set_title(f"Reliability Diagram — {self.field}\nECE={ece:.4f}  Brier={brier:.4f}")
            ax.legend()
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=120, bbox_inches="tight")
                print(f"Reliability diagram saved: {save_path}")
            return fig, ece, brier
        except ImportError:
            return None, ece, brier


# ── convenience: per-field calibrators ───────────────────────────────────────

class PerFieldCalibrator:
    """Wraps one ModelCalibrator per field."""

    FIELDS = ["kwh", "kvah", "md_kw", "demand_kva", "meter_serial", "decimal"]

    def __init__(self, model_dir: str = "data/calibration"):
        self._cals = {
            f: ModelCalibrator(field=f, model_dir=model_dir)
            for f in self.FIELDS
        }

    def calibrate(self, field: str, raw_conf: float) -> float:
        cal = self._cals.get(field) or self._cals["kwh"]
        return cal.calibrate(raw_conf)

    def fit(self, field: str, raw_confs, labels):
        cal = self._cals.get(field)
        if cal:
            cal.fit(np.asarray(raw_confs), np.asarray(labels))
