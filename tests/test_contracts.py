"""Interface-contract tests. These check the validators only; no detector is implemented or simulated here."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intruder.contracts import (CLASSES, DETECTORS, MODEL_INPUT_COLUMNS, PREDICTION_COLUMNS, PROGRESSION_COLUMNS,
                                STAGES, ContractError, check_score_direction, validate_explanations,
                                validate_predictions, validate_progression, validate_stage_signals,
                                validate_window_scores)
from intruder.features import FEATURES

ROOT = Path(__file__).parents[1]


def test_stage_and_class_vocabulary_is_fixed():
    assert STAGES == ("RECON", "CRED_ATTACK", "COMPROMISE", "PRIV_ESC", "PERSISTENCE", "EXFIL")
    assert CLASSES == ("Normal", "Suspicious", "Failed attack", "Intrusion")
    assert DETECTORS == ("A_anomaly_only", "B_stage_count", "C_progression")
    assert MODEL_INPUT_COLUMNS == FEATURES


def test_contract_document_lists_every_column_and_stage():
    doc = (ROOT / "docs" / "CONTRACTS.md").read_text(encoding="utf-8")
    for name in [*FEATURES, *STAGES, *PREDICTION_COLUMNS, *PROGRESSION_COLUMNS]:
        assert f"`{name}`" in doc, name


def test_detection_package_does_not_import_evaluation():
    for path in (ROOT / "intruder").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import evaluation" not in text and "from evaluation" not in text, path.name


# --- window scores / direction ----------------------------------------------------------------

def scores(**overrides):
    base = {"entity": ["10.0.1.5", "10.0.1.5"], "window_start": ["2026-10-04T10:00:00", "2026-10-04T10:05:00"],
            "anomaly_score": [0.1, 0.9]}
    return pd.DataFrame({**base, **overrides})


def test_window_scores_accept_unit_interval():
    out = validate_window_scores(scores())
    assert out.anomaly_score.between(0, 1).all()


@pytest.mark.parametrize("bad", [
    {"anomaly_score": [0.1, 1.5]},
    {"anomaly_score": [-0.2, 0.5]},
    {"anomaly_score": [np.nan, 0.5]},
    {"window_start": ["2026-10-04T10:00:00", "2026-10-04T10:00:00"]},  # duplicate key
])
def test_window_scores_reject_bad_values(bad):
    with pytest.raises(ContractError):
        validate_window_scores(scores(**bad))


def feature_frame(failed_logins):
    frame = pd.DataFrame(0.0, index=range(len(failed_logins)), columns=FEATURES)
    frame["failed_logins"] = failed_logins
    return frame


def test_score_direction_check_catches_an_inverted_scorer():
    # Test doubles for the direction check only - not detectors.
    def increases_with_failures(frame):
        return (frame.failed_logins / 100).clip(0, 1).to_numpy()

    def inverted(frame):
        return 1 - increases_with_failures(frame)

    typical, atypical = feature_frame([0, 1, 2]), feature_frame([80, 90])
    low, high = check_score_direction(increases_with_failures, typical, atypical)
    assert high > low
    with pytest.raises(ContractError, match="higher must mean more anomalous"):
        check_score_direction(inverted, typical, atypical)
    with pytest.raises(ContractError, match="within"):
        check_score_direction(lambda f: f.failed_logins.to_numpy(), typical, atypical)  # unnormalised


# --- explanations / stage signals -------------------------------------------------------------

def test_explanations_must_reference_known_features_and_positive_scale():
    good = pd.DataFrame({"entity": ["10.0.1.5"], "window_start": ["2026-10-04T10:00:00"],
                         "feature": ["failed_logins"], "value": [12.0], "baseline_median": [0.0],
                         "baseline_scale": [1.0], "deviation": [12.0]})
    validate_explanations(good)
    with pytest.raises(ContractError):
        validate_explanations(good.assign(feature="label"))
    with pytest.raises(ContractError):
        validate_explanations(good.assign(baseline_scale=0.0))


def test_stage_signals_need_known_stage_event_id_and_evidence():
    good = pd.DataFrame({"entity": ["203.0.113.10"], "ts": ["2026-10-04T10:02:00Z"], "stage": ["RECON"],
                         "eid": [17], "evidence": ["events.csv line 19"]})
    assert validate_stage_signals(good).eid.iloc[0] == 17
    for bad in ({"stage": "PORT_SCAN"}, {"eid": -1}, {"evidence": ""}, {"ts": "yesterday"}):
        with pytest.raises(ContractError):
            validate_stage_signals(good.assign(**bad))


# --- progression output -----------------------------------------------------------------------

def progression(**overrides):
    row = {"entity": "203.0.113.10", "day": "2026-10-04", "stages_detected": "RECON|CRED_ATTACK|COMPROMISE",
           "stage_times": "2026-10-04T10:02:00Z|2026-10-04T10:05:00Z|2026-10-04T10:07:00Z",
           "stages_completed": 3, "current_stage": "COMPROMISE", "next_expected_stage": "PRIV_ESC",
           "progression_score": 72.0, "first_warning_ts": "2026-10-04T10:02:00Z"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_partial_progression_row_is_valid():
    out = validate_progression(progression())
    assert out.first_warning_ts.iloc[0] == pd.Timestamp("2026-10-04 10:02")


def test_empty_and_final_stage_rows_are_valid():
    validate_progression(progression(stages_detected="", stage_times="", stages_completed=0, current_stage="",
                                     next_expected_stage="", progression_score=0.0, first_warning_ts=""))
    validate_progression(progression(stages_detected="COMPROMISE|EXFIL",
                                     stage_times="2026-10-04T10:07:00Z|2026-10-04T10:15:00Z", stages_completed=2,
                                     current_stage="EXFIL", next_expected_stage=""))


def test_out_of_order_observations_are_representable():
    # Reverse canonical order observed in time: valid representation; how it is scored is the detector's job.
    validate_progression(progression(stages_detected="COMPROMISE|RECON|CRED_ATTACK"))


@pytest.mark.parametrize("bad", [
    {"current_stage": "CRED_ATTACK"},                                     # not the most advanced stage
    {"next_expected_stage": "EXFIL"},                                     # must follow current_stage
    {"stages_completed": 2},                                              # must count distinct stages
    {"stage_times": "2026-10-04T10:05:00Z|2026-10-04T10:02:00Z|2026-10-04T10:07:00Z"},  # not chronological
    {"stage_times": "2026-10-04T10:02:00Z|2026-10-04T10:05:00Z"},          # misaligned
    {"stages_detected": "RECON|LATERAL|COMPROMISE"},                       # unknown stage
    {"progression_score": 150.0},
    {"first_warning_ts": "soon"},
])
def test_progression_contract_violations(bad):
    with pytest.raises(ContractError):
        validate_progression(progression(**bad))


def test_repeated_stage_counts_once():
    validate_progression(progression(
        stages_detected="CRED_ATTACK|CRED_ATTACK|COMPROMISE", stages_completed=2,
        stage_times="2026-10-04T10:02:00Z|2026-10-04T10:05:00Z|2026-10-04T10:07:00Z"))


# --- predictions ------------------------------------------------------------------------------

def predictions(**overrides):
    row = {"split": "test_id", "detector": "C_progression", "entity": "203.0.113.10", "day": "2026-10-04",
           "risk_score": 88.0, "predicted_class": "Intrusion", "alert": "True",
           "first_warning_ts": "2026-10-04T10:02:00Z", "current_stage": "COMPROMISE", "stages_completed": 3,
           "explanation": "example"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_predictions_are_normalised():
    out = validate_predictions(predictions())
    assert out.alert.iloc[0] is np.True_ or out.alert.iloc[0] is True
    normal = validate_predictions(predictions(predicted_class="Normal", alert="False", first_warning_ts="",
                                              current_stage="", stages_completed=0, risk_score=3))
    assert not normal.alert.iloc[0]


@pytest.mark.parametrize("bad", [
    {"alert": "False"},                         # inconsistent with predicted_class
    {"alert": "maybe"},
    {"first_warning_ts": ""},                   # alerts need a first warning time
    {"predicted_class": "Critical"},
    {"detector": "D_magic"},
    {"split": "holdout"},
    {"risk_score": 101},
    {"stages_completed": 7},
    {"current_stage": "PORT_SCAN"},
    {"day": "04/10/2026x"},
])
def test_prediction_contract_violations(bad):
    with pytest.raises(ContractError):
        validate_predictions(predictions(**bad))


def test_duplicate_predictions_rejected():
    with pytest.raises(ContractError, match="duplicate"):
        validate_predictions(pd.concat([predictions(), predictions()]))
