from __future__ import annotations

import pandas as pd
import pytest

from intruder.contracts import DETECTORS, PREDICTION_COLUMNS, validate_predictions
from intruder.prediction import build_all_detectors, build_predictions


def anomaly_rows():
    return pd.DataFrame([
        {"entity": "10.0.0.50", "window_start": "2026-10-04 10:00:00", "anomaly_score": 0.10},
        {"entity": "10.0.0.60", "window_start": "2026-10-04 10:00:00", "anomaly_score": 0.90},
    ])


def progression_rows():
    return pd.DataFrame([
        {
            "entity": "10.0.0.50",
            "day": "2026-10-04",
            "stages_detected": "",
            "stage_times": "",
            "stages_completed": 0,
            "current_stage": "",
            "next_expected_stage": "",
            "progression_score": 0,
            "first_warning_ts": "",
        },
        {
            "entity": "10.0.0.60",
            "day": "2026-10-04",
            "stages_detected": "RECON|CRED_ATTACK|COMPROMISE",
            "stage_times": (
                "2026-10-04T10:00:00|"
                "2026-10-04T10:03:00|"
                "2026-10-04T10:05:00"
            ),
            "stages_completed": 3,
            "current_stage": "COMPROMISE",
            "next_expected_stage": "PRIV_ESC",
            "progression_score": 50.0,
            "first_warning_ts": "2026-10-04T10:00:00",
        },
    ])


def test_low_anomaly_is_normal():
    result = build_predictions(
        anomaly_rows().iloc[[0]],
        progression_rows().iloc[[0]],
        split="test_id",
        detector="A_anomaly_only",
    )
    row = result.iloc[0]
    assert row["predicted_class"] == "Normal"
    assert bool(row["alert"]) is False
    assert row["current_stage"] == ""
    assert row["stages_completed"] == 0


def test_high_anomaly_is_suspicious_for_anomaly_only():
    result = build_predictions(
        anomaly_rows().iloc[[1]],
        progression_rows().iloc[[1]],
        split="test_id",
        detector="A_anomaly_only",
    )
    row = result.iloc[0]
    assert row["predicted_class"] == "Suspicious"
    assert bool(row["alert"]) is True
    assert row["risk_score"] == 90.0
    assert row["current_stage"] == ""
    assert row["stages_completed"] == 0


def test_stage_count_can_classify_failed_attack():
    anomaly = pd.DataFrame([{
        "entity": "10.0.0.70",
        "window_start": "2026-10-04 10:00:00",
        "anomaly_score": 0.60,
    }])
    progression = pd.DataFrame([{
        "entity": "10.0.0.70",
        "day": "2026-10-04",
        "stages_detected": "RECON|CRED_ATTACK",
        "stage_times": "2026-10-04T10:00:00|2026-10-04T10:03:00",
        "stages_completed": 2,
        "current_stage": "CRED_ATTACK",
        "next_expected_stage": "COMPROMISE",
        "progression_score": 33.33,
        "first_warning_ts": "2026-10-04T10:00:00",
    }])
    signals = pd.DataFrame([
        {"entity": "10.0.0.70", "ts": "2026-10-04 10:00:00", "stage": "RECON", "eid": 1, "evidence": "e"},
        {"entity": "10.0.0.70", "ts": "2026-10-04 10:03:00", "stage": "CRED_ATTACK", "eid": 2, "evidence": "e"},
    ])
    result = build_predictions(
        anomaly, progression, split="test_id", detector="B_stage_count", stage_signals=signals
    )
    assert result.iloc[0]["predicted_class"] == "Failed attack"


def test_progression_can_classify_intrusion():
    result = build_predictions(
        anomaly_rows().iloc[[1]],
        progression_rows().iloc[[1]],
        split="test_id",
        detector="C_progression",
    )
    row = result.iloc[0]
    assert row["predicted_class"] == "Intrusion"
    assert row["current_stage"] == "COMPROMISE"
    assert row["stages_completed"] == 3
    assert pd.Timestamp(row["first_warning_ts"]) == pd.Timestamp("2026-10-04T10:00:00")


def test_prediction_columns_are_exact():
    result = build_predictions(
        anomaly_rows().iloc[[0]],
        progression_rows().iloc[[0]],
        split="val",
        detector="A_anomaly_only",
    )
    assert list(result.columns) == PREDICTION_COLUMNS


def test_risk_is_within_contract_range():
    result = build_all_detectors(
        anomaly_rows(), progression_rows(), split="test_id"
    )
    assert result["risk_score"].between(0, 100).all()


def test_alert_matches_class():
    result = build_all_detectors(
        anomaly_rows(), progression_rows(), split="test_id"
    )
    assert (result["alert"] == (result["predicted_class"] != "Normal")).all()


def test_every_alert_has_warning_timestamp():
    result = build_all_detectors(
        anomaly_rows(), progression_rows(), split="test_id"
    )
    alerts = result[result["alert"]]
    assert alerts["first_warning_ts"].astype(str).str.strip().ne("").all()


def test_anomaly_only_leaves_stage_fields_empty():
    result = build_predictions(
        anomaly_rows(),
        progression_rows(),
        split="test_id",
        detector="A_anomaly_only",
    )
    assert result["current_stage"].eq("").all()
    assert result["stages_completed"].eq(0).all()


def test_progression_detector_carries_progression_fields():
    result = build_predictions(
        anomaly_rows().iloc[[1]],
        progression_rows().iloc[[1]],
        split="test_id",
        detector="C_progression",
    )
    row = result.iloc[0]
    assert row["current_stage"] == "COMPROMISE"
    assert row["stages_completed"] == 3


def test_predictions_are_deterministic():
    a = build_all_detectors(anomaly_rows(), progression_rows(), split="test_id")
    b = build_all_detectors(anomaly_rows(), progression_rows(), split="test_id")
    pd.testing.assert_frame_equal(a, b)


def test_generated_predictions_pass_validator():
    result = build_all_detectors(
        anomaly_rows(), progression_rows(), split="test_id"
    )
    checked = validate_predictions(result)
    assert len(checked) == 6


def test_missing_progression_still_allows_anomaly_prediction():
    result = build_predictions(
        anomaly_rows().iloc[[1]],
        pd.DataFrame(),
        split="test_ood",
        detector="A_anomaly_only",
    )
    assert result.iloc[0]["predicted_class"] == "Suspicious"


def test_invalid_detector_is_rejected():
    with pytest.raises(ValueError):
        build_predictions(
            anomaly_rows(), progression_rows(),
            split="test_id", detector="bad_detector"
        )


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        build_predictions(
            anomaly_rows(), progression_rows(),
            split="test", detector="A_anomaly_only"
        )
