"""
Isolation Forest anomaly detector.

This module is intentionally independent of ground truth and evaluation code.
It receives behavioural feature windows and produces a normalized anomaly score
where a higher value means more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import FEATURES, model_matrix


@dataclass
class IsolationForestDetector:
    """
    Wrapper around sklearn's IsolationForest.

    The wrapper keeps the project-level contract simple:

    - input: behavioural feature windows
    - output: anomaly score in [0, 1]
    - higher score: more anomalous
    - no ground-truth information is used
    """

    random_state: int = 42
    contamination: str | float = "auto"
    threshold: float = 0.5
    n_estimators: int = 200

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        if isinstance(self.contamination, float):
            if not 0.0 < self.contamination <= 0.5:
                raise ValueError(
                    "float contamination must be greater than 0 and at most 0.5"
                )

        self._model: IsolationForest | None = None
        self._feature_names = list(FEATURES)
        # Raw anomaly-score range on the training windows, fixed at fit time.
        self._train_score_min: float | None = None
        self._train_score_max: float | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the underlying model has been fitted."""
        return self._model is not None

    @property
    def feature_names(self) -> list[str]:
        """Return the feature names used by the model."""
        return list(self._feature_names)

    def _validate_features(self, features: pd.DataFrame) -> None:
        """Validate the feature-frame interface without modifying it."""
        missing = [name for name in FEATURES if name not in features.columns]

        if missing:
            raise ValueError(
                "Missing required feature columns: "
                + ", ".join(missing)
            )

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Convert a feature frame to the full FEATURES model matrix."""
        self._validate_features(features)

        return model_matrix(features)

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        """
        Fit the anomaly detector.

        `features` should contain benign training windows only. Ground truth
        labels are deliberately not accepted or consulted here.
        """
        self._validate_features(features)

        matrix = model_matrix(features)

        if len(matrix) < 2:
            raise ValueError("At least two feature windows are required to fit")

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

        self._model.fit(matrix)

        # Normalisation range comes from the training windows only, so every
        # later window is scored on the same fixed scale.
        train_raw = self._raw_anomaly_score(matrix)
        self._train_score_min = float(np.min(train_raw))
        self._train_score_max = float(np.max(train_raw))

        return self

    def _raw_anomaly_score(self, matrix: np.ndarray) -> np.ndarray:
        """
        Return sklearn's anomaly score with its sign convention converted.

        sklearn's score_samples() gives larger values to more normal samples.
        We negate it so larger values mean more anomalous.
        """
        if self._model is None:
            raise RuntimeError("Detector must be fitted before scoring")

        return -self._model.score_samples(matrix)

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Normalize raw anomaly scores to [0, 1] using the training-time range.

        0 corresponds to the least anomalous training window and 1 to the most
        anomalous one; scores outside that range are clipped. The scale is fixed
        at fit time, so a window's score does not depend on what else is in the
        scoring batch. This is a ranking-oriented normalization, not a
        probability.

        If every training window had the same raw score, the range is empty:
        scores below it map to 0, equal to it to 0.5, and above it to 1.
        """
        if self._train_score_min is None or self._train_score_max is None:
            raise RuntimeError("Detector must be fitted before scoring")

        raw_scores = np.asarray(raw_scores, dtype=float)

        if raw_scores.size == 0:
            return raw_scores

        minimum = self._train_score_min
        maximum = self._train_score_max

        if np.isclose(minimum, maximum):
            same = np.isclose(raw_scores, maximum)
            normalized = np.full(raw_scores.shape, 0.5, dtype=float)
            normalized[(raw_scores > maximum) & ~same] = 1.0
            normalized[(raw_scores < minimum) & ~same] = 0.0
            return normalized

        normalized = (raw_scores - minimum) / (maximum - minimum)

        return np.clip(normalized, 0.0, 1.0)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return normalized anomaly scores for a feature frame.

        Higher values indicate greater behavioural anomaly.
        """
        matrix = self._matrix(features)
        raw_scores = self._raw_anomaly_score(matrix)

        return self._normalize_scores(raw_scores)

    def score_windows(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return project-contract window scores.

        Columns:
            entity
            window_start
            anomaly_score
        """
        scores = self.score(features)

        return pd.DataFrame(
            {
                "entity": features["entity"].to_numpy(copy=True),
                "window_start": features["window_start"].to_numpy(copy=True),
                "anomaly_score": scores,
            }
        )

    def is_anomalous(self, scores: Iterable[float]) -> np.ndarray:
        """
        Convert anomaly scores into deterministic alert decisions.
        """
        values = np.asarray(list(scores), dtype=float)

        if np.any(~np.isfinite(values)):
            raise ValueError("scores must contain only finite values")

        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("scores must be between 0 and 1")

        return values >= self.threshold

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Score windows and add an anomaly alert column.
        """
        result = self.score_windows(features)

        result["alert"] = self.is_anomalous(
            result["anomaly_score"].to_numpy()
        )

        return result