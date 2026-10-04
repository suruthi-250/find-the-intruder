import numpy as np
import pandas as pd
import pytest

from intruder.features import FEATURES
from intruder.model import IsolationForestDetector


def feature_frame(n=20):
    """Small deterministic feature frame for model tests."""
    rows = []

    for i in range(n):
        row = {
            "entity": f"10.0.0.{10 + i}",
            "window_start": pd.Timestamp("2026-01-01 10:00:00") + pd.Timedelta(minutes=5 * i),
        }

        for feature in FEATURES:
            row[feature] = 0.0

        # Small benign variation, like real background windows: each window
        # holds one ordinary event. Web requests are the most common; a
        # connection, a successful login, or an occasional single mistyped
        # password are rarer.
        kind = i % 5
        if kind in (0, 1):
            row["web_requests"] = 1.0
        elif kind == 2:
            row["distinct_ports"] = 1.0
        elif kind == 3:
            row["successful_logins"] = 1.0
            row["distinct_users"] = 1.0
        else:
            row["failed_logins"] = 1.0
            row["distinct_users"] = 1.0
            row["failure_ratio"] = 1.0

        row["event_count"] = 1.0
        row["event_rate"] = 0.2

        rows.append(row)

    return pd.DataFrame(rows)


def test_model_can_fit_on_feature_frame():
    features = feature_frame()

    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
    )

    detector.fit(features)

    assert detector.is_fitted


def test_model_returns_one_score_per_window():
    features = feature_frame()

    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
    )

    detector.fit(features)

    scores = detector.score_windows(features)

    assert len(scores) == len(features)
    assert list(scores.columns) == [
        "entity",
        "window_start",
        "anomaly_score",
    ]


def test_anomaly_scores_are_between_zero_and_one():
    features = feature_frame()

    # Add an intentionally unusual window.
    for feature in [
        "failed_logins",
        "distinct_ports",
        "denied_connections",
        "event_count",
    ]:
        features.loc[19, feature] = 100.0

    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
    )

    detector.fit(features.iloc[:15])

    scores = detector.score_windows(features)

    assert scores["anomaly_score"].between(0.0, 1.0).all()


def test_more_anomalous_window_gets_higher_score():
    features = feature_frame(21)

    # Training data remains normal.
    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
    )

    detector.fit(features.iloc[:15])

    normal = features.iloc[[15]].copy()
    anomalous = features.iloc[[20]].copy()

    anomalous.loc[:, "failed_logins"] = 100.0
    anomalous.loc[:, "distinct_ports"] = 100.0
    anomalous.loc[:, "denied_connections"] = 100.0
    anomalous.loc[:, "event_count"] = 100.0
    anomalous.loc[:, "event_rate"] = 20.0

    normal_score = detector.score_windows(normal)["anomaly_score"].iloc[0]
    anomalous_score = detector.score_windows(anomalous)["anomaly_score"].iloc[0]

    assert anomalous_score > normal_score


def test_threshold_classification_is_deterministic():
    features = feature_frame()

    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
        threshold=0.5,
    )

    detector.fit(features)

    scores = detector.score_windows(features)

    expected = scores["anomaly_score"] >= 0.5

    assert np.array_equal(
        detector.is_anomalous(scores["anomaly_score"].to_numpy()),
        expected.to_numpy(),
    )


def test_model_does_not_modify_input_features():
    features = feature_frame()
    original = features.copy(deep=True)

    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
    )

    detector.fit(features)
    detector.score_windows(features)

    pd.testing.assert_frame_equal(features, original)


def test_model_rejects_scoring_before_fit():
    features = feature_frame()

    detector = IsolationForestDetector(
        random_state=42,
        contamination="auto",
    )

    with pytest.raises(RuntimeError):
        detector.score_windows(features)