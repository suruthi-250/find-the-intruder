"""Evaluation-harness tests on small hand-made prediction and ground-truth tables."""
import json

import pandas as pd
import pytest

from evaluation import run
from evaluation.ground_truth import GT_COLUMNS, validate_ground_truth
from evaluation.metrics import MAJORITY, evaluate, results_table
from intruder.contracts import ContractError

DAY = "2026-10-04"


def gt_row(entity, label, trap="", start="", end="", stages="", split="test_id", scenario=None):
    return {"split": split, "scenario_id": scenario or f"{split}-{entity}", "entity_id": entity, "day": DAY,
            "label": label, "trap_type": trap, "attack_start": start, "attack_end": end,
            "expected_stages": stages, "profile": "shifted" if split == "test_ood" else "base"}


def truth_rows():
    return [
        gt_row("203.0.113.10", "successful_intrusion", start=f"{DAY}T10:00:00Z", end=f"{DAY}T11:00:00Z",
               stages="RECON|CRED_ATTACK|COMPROMISE|PRIV_ESC"),
        gt_row("198.51.100.20", "failed_attack", start=f"{DAY}T12:00:00Z", end=f"{DAY}T12:30:00Z",
               stages="RECON|CRED_ATTACK"),
        gt_row("10.0.1.5", "benign_trap", trap="typo_login"),
        gt_row("10.0.3.7", "benign_trap", trap="internal_backup"),
        gt_row("10.0.1.6", "benign"),
        gt_row("192.0.2.30", "successful_intrusion", start=f"{DAY}T09:00:00Z", end=f"{DAY}T09:40:00Z",
               stages="CRED_ATTACK|COMPROMISE", split="test_ood"),
    ]


def truth():
    return validate_ground_truth(pd.DataFrame(truth_rows(), columns=GT_COLUMNS))


def pred(entity, cls, warning="", split="test_id", detector="C_progression"):
    return {"split": split, "detector": detector, "entity": entity, "day": DAY, "risk_score": 0 if cls == "Normal" else 80,
            "predicted_class": cls, "alert": str(cls != "Normal"), "first_warning_ts": warning,
            "current_stage": "", "stages_completed": 0, "explanation": ""}


def predictions():
    return pd.DataFrame([
        pred("203.0.113.10", "Intrusion", f"{DAY}T10:05:00Z"),     # TP
        pred("198.51.100.20", "Normal"),                            # FN
        pred("10.0.1.5", "Normal"),                                 # TN (trap A)
        pred("10.0.3.7", "Suspicious", f"{DAY}T02:00:00Z"),         # FP (trap B)
        pred("10.0.1.6", "Normal"),                                 # TN
        pred("10.0.1.9", "Normal"),                                 # not in ground truth -> benign TN
        pred("10.0.1.6", "Normal", split="test_ood"),               # ood: intrusion row missing -> FN
    ])


# --- ground truth validation ------------------------------------------------------------------

def test_valid_ground_truth_is_normalised():
    t = truth()
    assert t.attack_start.iloc[0] == pd.Timestamp(f"{DAY} 10:00")
    assert str(t.day.iloc[0]) == DAY


@pytest.mark.parametrize("index, change", [
    (4, {"label": "mostly_benign"}),
    (4, {"trap_type": "typo_login"}),                                    # trap type on a non-trap row
    (2, {"trap_type": ""}),                                              # trap row without type
    (1, {"expected_stages": "RECON|CRED_ATTACK|COMPROMISE"}),            # failed attack cannot compromise
    (0, {"expected_stages": "RECON|CRED_ATTACK"}),                       # intrusion must compromise
    (0, {"expected_stages": "CRED_ATTACK|RECON|COMPROMISE"}),            # non-canonical order
    (0, {"attack_start": ""}),
    (0, {"attack_end": f"{DAY}T09:00:00Z"}),                             # ends before it starts
    (0, {"attack_end": "2026-10-05T01:00:00Z"}),                         # crosses midnight
    (4, {"attack_start": f"{DAY}T10:00:00Z"}),                           # benign rows carry no attack times
    (4, {"entity_id": "8.8.8.8"}),                                       # non-synthetic address
    (0, {"split": "train"}),                                             # train must be benign only
])
def test_ground_truth_violations(index, change):
    rows = truth_rows()
    rows[index] = {**rows[index], **change}
    with pytest.raises(ContractError):
        validate_ground_truth(pd.DataFrame(rows, columns=GT_COLUMNS))


def test_duplicate_entity_day_rejected():
    rows = truth_rows()
    with pytest.raises(ContractError, match="duplicate"):
        validate_ground_truth(pd.DataFrame(rows + [rows[4]], columns=GT_COLUMNS))


# --- metrics ----------------------------------------------------------------------------------

def test_binary_trap_and_label_metrics():
    m = evaluate(predictions(), truth())["detectors"]["C_progression"]["splits"]["test_id"]
    assert {k: m["binary"][k] for k in ("tp", "fp", "fn", "tn")} == {"tp": 1, "fp": 1, "fn": 1, "tn": 3}
    assert (m["binary"]["precision"], m["binary"]["recall"], m["binary"]["f1"], m["binary"]["fpr"]) == (0.5, 0.5, 0.5, 0.25)
    assert m["traps"]["internal_backup"] == {"n": 1, "false_positives": 1, "fp_rate": 1.0}
    assert m["traps"]["typo_login"] == {"n": 1, "false_positives": 0, "fp_rate": 0.0}
    assert m["traps"]["noisy_benign"]["fp_rate"] is None                 # no examples -> undefined, not zero
    assert m["per_label"]["successful_intrusion"]["alert_rate"] == 1.0
    assert m["per_label"]["failed_attack"]["alert_rate"] == 0.0


def test_latency_and_lead_time():
    timing = evaluate(predictions(), truth())["detectors"]["C_progression"]["splits"]["test_id"]["timing"]
    assert timing["all_attacks"]["latency"]["median_s"] == 300.0       # 10:05 warning - 10:00 start
    assert timing["all_attacks"]["lead_time"]["median_s"] == 3300.0    # 11:00 end - 10:05 warning
    assert timing["all_attacks"]["warned_before_end_rate"] == 1.0
    assert timing["failed_attack"]["latency"]["n"] == 0                 # missed attacks have no latency


def test_four_class_confusion():
    fc = evaluate(predictions(), truth())["detectors"]["C_progression"]["splits"]["test_id"]["four_class"]
    assert fc["confusion"]["benign_trap"] == {"Normal": 1, "Suspicious": 1, "Failed attack": 0, "Intrusion": 0}
    assert fc["accuracy_by_label"] == {"benign": 1.0, "benign_trap": 0.5, "failed_attack": 0.0,
                                       "successful_intrusion": 1.0}


def test_missing_attack_prediction_is_a_false_negative_and_ood_gap():
    entry = evaluate(predictions(), truth())["detectors"]["C_progression"]
    ood = entry["splits"]["test_ood"]
    assert ood["missing_predictions"] == 1
    assert ood["binary"]["fn"] == 1 and ood["binary"]["f1"] == 0.0
    assert entry["generalisation_gap_f1"] == 0.5


def test_majority_baseline_never_alerts():
    m = evaluate(predictions(), truth())["detectors"][MAJORITY]["splits"]["test_id"]["binary"]
    assert (m["tp"], m["fp"], m["precision"], m["recall"], m["fpr"]) == (0, 0, None, 0.0, 0.0)


def test_detectors_are_compared_on_the_same_entity_days():
    preds = pd.concat([predictions(), predictions().assign(detector="A_anomaly_only", predicted_class="Normal",
                                                           alert="False")])
    table = results_table(evaluate(preds, truth()))
    sizes = {(r.detector, r.split) for r in table.itertuples()}
    assert {d for d, _ in sizes} == {"A_anomaly_only", "C_progression", MAJORITY}
    assert table.groupby("split").n_entity_days.nunique().eq(1).all()


def test_cli_writes_metrics(tmp_path):
    data = tmp_path / "data"
    rows = pd.DataFrame(truth_rows(), columns=GT_COLUMNS)
    for split in ("test_id", "test_ood"):
        (data / split).mkdir(parents=True)
        rows[rows.split == split].to_csv(data / split / "ground_truth.csv", index=False)
    predictions().to_csv(tmp_path / "predictions.csv", index=False)
    run.main(["--predictions", str(tmp_path / "predictions.csv"), "--data", str(data), "--out", str(tmp_path / "out")])
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["detectors"]["C_progression"]["splits"]["test_id"]["binary"]["f1"] == 0.5
    assert (tmp_path / "out" / "results_table.csv").exists()
