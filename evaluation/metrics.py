"""Compare detectors against ground truth at the entity-day level.

Positive class (attack)  = failed_attack + successful_intrusion
Negative class (benign)  = benign + benign_trap
A detector "alerts" on an entity-day when its predicted_class is not Normal.

Entity-days a detector scored but ground truth does not list are benign. Entity-days ground truth lists but a
detector did not score count as not alerted (and are reported as missing_predictions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from intruder.contracts import CLASSES, validate_predictions

from .ground_truth import ATTACK_LABELS, LABELS, TRAP_TYPES

MAJORITY = "0_majority"
CORRECT_CLASS = {"benign": "Normal", "benign_trap": "Normal", "failed_attack": "Failed attack",
                 "successful_intrusion": "Intrusion"}


def _ratio(num, den):
    return None if den == 0 else round(float(num) / float(den), 4)


def _summary(seconds: pd.Series) -> dict:
    s = seconds.dropna()
    if s.empty:
        return {"n": 0, "median_s": None, "p90_s": None, "mean_s": None, "min_s": None, "max_s": None}
    return {"n": int(len(s)), "median_s": round(float(s.median()), 1), "p90_s": round(float(s.quantile(0.9)), 1),
            "mean_s": round(float(s.mean()), 1), "min_s": round(float(s.min()), 1), "max_s": round(float(s.max()), 1)}


def join(predictions: pd.DataFrame, truth: pd.DataFrame, split: str, detector: str) -> pd.DataFrame:
    """Outer-join one detector's predictions with ground truth for one split on (entity, day)."""
    p = predictions[(predictions.split == split) & (predictions.detector == detector)]
    p = p[["entity", "day", "predicted_class", "alert", "first_warning_ts"]]
    t = truth[truth.split == split].rename(columns={"entity_id": "entity"})
    t = t[["entity", "day", "label", "trap_type", "attack_start", "attack_end"]]
    j = p.merge(t, on=["entity", "day"], how="outer", indicator=True)
    j["missing_prediction"] = j.pop("_merge") == "right_only"
    j["label"] = j["label"].fillna("benign")
    j["trap_type"] = j["trap_type"].fillna("")
    j["predicted_class"] = j["predicted_class"].fillna("Normal")
    j["alert"] = j["alert"].map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
    return j


def binary_metrics(j: pd.DataFrame) -> dict:
    actual, alert = j.label.isin(ATTACK_LABELS), j.alert
    tp, fp = int((actual & alert).sum()), int((~actual & alert).sum())
    fn, tn = int((actual & ~alert).sum()), int((~actual & ~alert).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "fpr": _ratio(fp, fp + tn)}


def per_label_detection(j: pd.DataFrame) -> dict:
    """Alert rate per ground-truth label (recall for attack labels, false-alarm rate for benign ones)."""
    return {label: {"n": int((j.label == label).sum()), "alert_rate": _ratio(j.alert[j.label == label].sum(),
                                                                             (j.label == label).sum())}
            for label in LABELS}


def trap_metrics(j: pd.DataFrame) -> dict:
    traps = j[j.label == "benign_trap"]
    return {trap: {"n": int((traps.trap_type == trap).sum()),
                   "false_positives": int(traps.alert[traps.trap_type == trap].sum()),
                   "fp_rate": _ratio(traps.alert[traps.trap_type == trap].sum(), (traps.trap_type == trap).sum())}
            for trap in TRAP_TYPES}


def timing_metrics(j: pd.DataFrame) -> dict:
    """Detection latency = first_warning_ts - attack_start (lower is better; negative means the first warning
    landed in a window that began before the labelled start).
    Lead time = attack_end - first_warning_ts (higher is better: how long before the attack finished it was flagged).
    Computed on detected attacks only."""
    hit = j[j.label.isin(ATTACK_LABELS) & j.alert & j.first_warning_ts.notna()]
    out = {}
    for name, rows in [("all_attacks", hit)] + [(label, hit[hit.label == label]) for label in ATTACK_LABELS]:
        latency = (rows.first_warning_ts - rows.attack_start).dt.total_seconds()
        lead = (rows.attack_end - rows.first_warning_ts).dt.total_seconds()
        out[name] = {"latency": _summary(latency), "lead_time": _summary(lead),
                     "warned_before_end_rate": _ratio((lead > 0).sum(), len(rows))}
    return out


def four_class(j: pd.DataFrame) -> dict:
    confusion = pd.crosstab(j.label, j.predicted_class).reindex(index=list(LABELS), columns=list(CLASSES),
                                                                fill_value=0)
    correct = j.predicted_class == j.label.map(CORRECT_CLASS)
    return {"confusion": {lab: {c: int(confusion.at[lab, c]) for c in CLASSES} for lab in LABELS},
            "accuracy_by_label": {lab: _ratio(correct[j.label == lab].sum(), (j.label == lab).sum()) for lab in LABELS},
            "note": "Suspicious counts as an alert in binary metrics but is never the 'correct' four-class answer."}


def majority_predictions(predictions: pd.DataFrame, truth: pd.DataFrame, splits) -> pd.DataFrame:
    """Baseline 0: predict Normal for every entity-day seen by any detector or listed in ground truth."""
    keys = pd.concat([predictions[["split", "entity", "day"]],
                      truth[truth.split.isin(splits)].rename(columns={"entity_id": "entity"})[["split", "entity", "day"]]])
    keys = keys[keys.split.isin(splits)].drop_duplicates()
    return keys.assign(detector=MAJORITY, predicted_class="Normal", alert=False, first_warning_ts=pd.NaT)


def evaluate(predictions: pd.DataFrame, truth: pd.DataFrame, splits=("test_id", "test_ood"),
             include_majority: bool = True) -> dict:
    """`truth` must already be validated (evaluation.ground_truth.validate_ground_truth)."""
    preds = validate_predictions(predictions)
    if include_majority:
        preds = pd.concat([preds, majority_predictions(preds, truth, splits)], ignore_index=True)
    results = {"splits": list(splits), "positive_labels": list(ATTACK_LABELS), "detectors": {}}
    for detector in sorted(preds.detector.unique()):
        per_split = {}
        for split in splits:
            j = join(preds, truth, split, detector)
            per_split[split] = {"n_entity_days": int(len(j)), "missing_predictions": int(j.missing_prediction.sum()),
                                "binary": binary_metrics(j), "per_label": per_label_detection(j),
                                "traps": trap_metrics(j), "timing": timing_metrics(j), "four_class": four_class(j)}
        entry = {"splits": per_split}
        if "test_id" in per_split and "test_ood" in per_split:
            f_id, f_ood = per_split["test_id"]["binary"]["f1"], per_split["test_ood"]["binary"]["f1"]
            entry["generalisation_gap_f1"] = None if f_id is None or f_ood is None else round(f_id - f_ood, 4)
        results["detectors"][detector] = entry
    return results


def results_table(results: dict) -> pd.DataFrame:
    """Flat detector x split table for the README and the dashboard."""
    rows = []
    for detector, entry in results["detectors"].items():
        for split, m in entry["splits"].items():
            b, timing = m["binary"], m["timing"]["all_attacks"]
            rows.append({"detector": detector, "split": split, "n_entity_days": m["n_entity_days"],
                         "precision": b["precision"], "recall": b["recall"],
                         "f1": b["f1"], "fpr": b["fpr"],
                         "trap_fp": sum(t["false_positives"] for t in m["traps"].values()),
                         "median_latency_s": timing["latency"]["median_s"],
                         "median_lead_time_s": timing["lead_time"]["median_s"],
                         "missing_predictions": m["missing_predictions"]})
    return pd.DataFrame(rows)
