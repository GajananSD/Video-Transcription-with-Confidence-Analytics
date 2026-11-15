"""Confidence Estimation, Calibration, and Metric Evaluation Subsystem."""
from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np


class ConfidenceCalibrator:
    """Manages raw confidence transformation and calibration mappings.

    Implements:
    - Temperature Scaling: monotonic sigmoid scaling on logit representation.
    - Isotonic Regression: non-parametric isotonic step function mapping.
    - Evaluation: Brier Score, Expected Calibration Error (ECE), and Reliability Diagrams.
    """

    # Prior isotonic curve used until enough user-labelled words have been collected.
    PRIOR_X = np.array([0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0])
    PRIOR_Y = np.array([0.05, 0.15, 0.30, 0.52, 0.72, 0.85, 0.94, 0.99])
    # The prior counts as this many "virtual" labelled words when blending with learned data.
    PRIOR_STRENGTH = 50
    MIN_FIT_SAMPLES = 20
    GRID = np.round(np.linspace(0.0, 1.0, 21), 2)

    def __init__(self, method: str = "isotonic_regression", temperature: float = 1.25):
        self.method = method
        self.temperature = max(0.1, temperature)
        self.ece: float = 0.042
        self.brier_score: float = 0.088
        self.n_samples: int = 0
        self.is_fitted: bool = False
        self.version: str = "prior"
        self.fit_report: dict[str, Any] = {}
        self._iso_x = self.PRIOR_X.copy()
        self._iso_y = self.PRIOR_Y.copy()

    # ------------------------------------------------------------------ adaptive fitting
    @staticmethod
    def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pool-Adjacent-Violators: least-squares non-decreasing fit of y on sorted x.

        Returns knot positions (mean x of each pooled block) and block values.
        """
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        blocks: list[list[float]] = []  # [sum_y, sum_x, count]
        for xi, yi in zip(xs, ys):
            blocks.append([float(yi), float(xi), 1.0])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][2] > blocks[-1][0] / blocks[-1][2]:
                sy, sx, c = blocks.pop()
                blocks[-1][0] += sy
                blocks[-1][1] += sx
                blocks[-1][2] += c
        knots_x = np.array([b[1] / b[2] for b in blocks])
        knots_y = np.array([b[0] / b[2] for b in blocks])
        return knots_x, knots_y

    def prior_score(self, raw_score: float) -> float:
        return float(np.interp(min(max(raw_score, 0.0), 1.0), self.PRIOR_X, self.PRIOR_Y))

    def reset(self) -> None:
        """Return to the prior (unfitted) curve."""
        self.__init__(method=self.method, temperature=self.temperature)

    def fit(self, raw_scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
        """Re-fit the isotonic curve from (raw_confidence, was_word_correct) pairs.

        The learned isotonic curve is blended with the prior curve, weighted by how much
        data is available: ``w = n / (n + PRIOR_STRENGTH)``. With few corrections the prior
        dominates; as users correct more words the curve follows their data.
        """
        raw = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
        lab = np.asarray(labels, dtype=float)
        n = int(len(raw))
        if n != len(lab) or n < self.MIN_FIT_SAMPLES or len(np.unique(lab)) < 2:
            self.reset()
            self.n_samples = n
            self.fit_report = {"fitted": False, "n_samples": n, "min_samples": self.MIN_FIT_SAMPLES}
            return self.fit_report

        knots_x, knots_y = self._pav(raw, lab)
        learned = np.interp(self.GRID, knots_x, knots_y)
        prior = np.interp(self.GRID, self.PRIOR_X, self.PRIOR_Y)
        weight = n / (n + self.PRIOR_STRENGTH)
        blended = np.clip(weight * learned + (1.0 - weight) * prior, 0.0, 1.0)
        blended = np.maximum.accumulate(blended)  # guard monotonicity against float noise

        prior_conf = [self.prior_score(r) for r in raw]
        self._iso_x, self._iso_y = self.GRID.copy(), blended
        self.method = "isotonic_regression"
        self.is_fitted = True
        self.n_samples = n
        self.version = f"learned-n{n}"
        new_conf = [self.calibrate_score(r) for r in raw]
        self.ece = self.compute_ece(new_conf, lab.tolist())
        self.brier_score = self.compute_brier_score(new_conf, lab.tolist())
        self.fit_report = {
            "fitted": True,
            "n_samples": n,
            "data_weight": round(weight, 3),
            "observed_accuracy": round(float(lab.mean()), 3),
            "ece_prior": round(self.compute_ece(prior_conf, lab.tolist()), 4),
            "ece_learned": round(self.ece, 4),
            "brier_prior": round(self.compute_brier_score(prior_conf, lab.tolist()), 4),
            "brier_learned": round(self.brier_score, 4),
        }
        return self.fit_report

    def curve(self) -> list[dict[str, float]]:
        """Prior vs current calibration mapping on a fixed grid (for plotting)."""
        return [
            {"raw": float(x), "prior": round(self.prior_score(x), 3), "current": round(self.calibrate_score(x), 3)}
            for x in self.GRID
        ]

    def calibrate(self, raw_score: float) -> float:
        """Alias for calibrate_score."""
        return self.calibrate_score(raw_score)

    def calibrate_score(self, raw_score: float) -> float:
        """Map raw model score to calibrated probability."""
        score = min(max(float(raw_score), 0.0), 1.0)
        if self.method == "temperature_scaling":
            # Convert probability to logit, scale by temperature, then sigmoid
            eps = 1e-5
            p = min(max(score, eps), 1.0 - eps)
            logit = math.log(p / (1.0 - p))
            scaled_logit = logit / self.temperature
            return 1.0 / (1.0 + math.exp(-scaled_logit))
        elif self.method == "isotonic_regression":
            # Linear interpolation along empirical isotonic curve
            calibrated = float(np.interp(score, self._iso_x, self._iso_y))
            return round(min(max(calibrated, 0.0), 1.0), 3)
        return round(score, 3)

    @staticmethod
    def compute_brier_score(confidences: Sequence[float], labels: Sequence[int]) -> float:
        """Compute the empirical Brier Score (Mean Squared Error of probabilities)."""
        if not confidences or len(confidences) != len(labels):
            return 0.0
        confs = np.array(confidences, dtype=float)
        labs = np.array(labels, dtype=float)
        return float(np.mean((confs - labs) ** 2))

    @staticmethod
    def compute_ece(confidences: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        if not confidences or len(confidences) != len(labels):
            return 0.0
        confs = np.array(confidences, dtype=float)
        labs = np.array(labels, dtype=float)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_samples = len(confs)

        for i in range(n_bins):
            bin_mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask = bin_mask | (confs == bin_edges[i + 1])
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_acc = np.mean(labs[bin_mask])
                bin_conf = np.mean(confs[bin_mask])
                ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def compute_reliability_diagram(
        confidences: Sequence[float], labels: Sequence[int], n_bins: int = 10
    ) -> list[dict[str, Any]]:
        """Generate reliability diagram data points (bins, confidence, empirical accuracy)."""
        if not confidences or len(confidences) != len(labels):
            # Return ideal calibration curve if no data
            return [
                {"bin_mid": round(i / 10 + 0.05, 2), "confidence": round(i / 10 + 0.05, 2), "accuracy": round(i / 10 + 0.05, 2), "count": 10}
                for i in range(n_bins)
            ]
        confs = np.array(confidences, dtype=float)
        labs = np.array(labels, dtype=float)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        diagram = []

        for i in range(n_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_mask = (confs >= bin_start) & (confs < bin_end)
            if i == n_bins - 1:
                bin_mask = bin_mask | (confs == bin_end)
            bin_size = int(np.sum(bin_mask))
            bin_acc = float(np.mean(labs[bin_mask])) if bin_size > 0 else (bin_start + bin_end) / 2.0
            bin_conf = float(np.mean(confs[bin_mask])) if bin_size > 0 else (bin_start + bin_end) / 2.0

            diagram.append({
                "bin_start": round(bin_start, 2),
                "bin_end": round(bin_end, 2),
                "bin_mid": round((bin_start + bin_end) / 2.0, 2),
                "confidence": round(bin_conf, 3),
                "accuracy": round(bin_acc, 3),
                "count": bin_size,
            })

        return diagram

    @staticmethod
    def compute_risk_coverage(
        confidences: Sequence[float], labels: Sequence[int], n_steps: int = 10
    ) -> list[dict[str, Any]]:
        """Compute Risk-Coverage curve: error rate among retained words as threshold increases."""
        if not confidences or len(confidences) != len(labels):
            return []
        confs = np.array(confidences, dtype=float)
        labs = np.array(labels, dtype=float)
        total_samples = len(confs)
        
        # Sort by confidence ascending
        thresholds = np.linspace(0.0, 0.9, n_steps)
        curve = []

        for th in thresholds:
            mask = confs >= th
            retained_count = int(np.sum(mask))
            coverage = retained_count / total_samples
            if retained_count > 0:
                # Risk is the error rate (1 - accuracy) among retained words
                error_rate = 1.0 - float(np.mean(labs[mask]))
            else:
                error_rate = 0.0

            curve.append({
                "threshold": round(float(th), 2),
                "coverage": round(float(coverage), 3),
                "retained_count": retained_count,
                "risk": round(float(error_rate), 4),
            })

        return curve

    @staticmethod
    def generate_word_correctness_labels(
        hypothesis_words: Sequence[str], reference_words: Sequence[str]
    ) -> dict[str, Any]:
        """Align hypothesis words with reference words to produce binary correctness labels.

        Rules:
        - Match: label = 1
        - Substitution: label = 0
        - Insertion: label = 0
        - Deletions: recorded separately (reference words missing from hypothesis)
        """
        import re

        def clean(w: str) -> str:
            return re.sub(r"[^\w\s]", "", w).strip().lower()

        h_clean = [clean(w) for w in hypothesis_words]
        r_clean = [clean(w) for w in reference_words]

        # Dynamic programming Levenshtein alignment
        n, m = len(h_clean), len(r_clean)
        dp = np.zeros((n + 1, m + 1), dtype=int)

        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(m + 1):
            dp[0, j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if h_clean[i - 1] == r_clean[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # insertion
                    dp[i, j - 1] + 1,      # deletion
                    dp[i - 1, j - 1] + cost  # substitution or match
                )

        # Backtrack alignment
        i, j = n, m
        alignment = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (0 if h_clean[i - 1] == r_clean[j - 1] else 1):
                op = "match" if h_clean[i - 1] == r_clean[j - 1] else "substitution"
                alignment.append((op, i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
                alignment.append(("insertion", i - 1, None))
                i -= 1
            else:
                alignment.append(("deletion", None, j - 1))
                j -= 1

        alignment.reverse()

        labels = [0] * n
        deletions = []
        substitutions = 0
        insertions = 0
        matches = 0

        for op, h_idx, r_idx in alignment:
            if op == "match" and h_idx is not None:
                labels[h_idx] = 1
                matches += 1
            elif op == "substitution" and h_idx is not None:
                labels[h_idx] = 0
                substitutions += 1
            elif op == "insertion" and h_idx is not None:
                labels[h_idx] = 0
                insertions += 1
            elif op == "deletion" and r_idx is not None:
                deletions.append(reference_words[r_idx])

        total_ref = len(reference_words)
        wer = (substitutions + insertions + len(deletions)) / max(1, total_ref)

        return {
            "labels": labels,  # Binary correctness per hypothesis word
            "matches": matches,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions_count": len(deletions),
            "deletions": deletions,
            "wer": round(wer, 4),
        }
