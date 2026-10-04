"""Interface contracts for the detection components: window scores, explanations, stage signals,
progression output and final predictions.

This module only validates. It contains no detection logic: the anomaly model, stage inference,
progression scoring and risk classification are implemented separately and must emit frames that pass
these checks. Validators return a normalised copy (parsed timestamps, booleans) or raise ContractError.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .features import FEATURES

STAGES = ("RECON", "CRED_ATTACK", "COMPROMISE", "PRIV_ESC", "PERSISTENCE", "EXFIL")
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
CLASSES = ("Normal", "Suspicious", "Failed attack", "Intrusion")
DETECTORS = ("A_anomaly_only", "B_stage_count", "C_progression")
SPLITS = ("train", "val", "test_id", "test_ood")

MODEL_INPUT_COLUMNS = list(FEATURES)
SCORE_COLUMNS = ["entity", "window_start", "anomaly_score"]
EXPLANATION_COLUMNS = ["entity", "window_start", "feature", "value", "baseline_median", "baseline_scale", "deviation"]
STAGE_SIGNAL_COLUMNS = ["entity", "ts", "stage", "eid", "evidence"]
PROGRESSION_COLUMNS = ["entity", "day", "stages_detected", "stage_times", "stages_completed", "current_stage",
                       "next_expected_stage", "progression_score", "first_warning_ts"]
PREDICTION_COLUMNS = ["split", "detector", "entity", "day", "risk_score", "predicted_class", "alert",
                      "first_warning_ts", "current_stage", "stages_completed", "explanation"]
MAX_REPORTED = 8  # problems listed per error message


class ContractError(ValueError):
    """A component produced output that does not match its interface contract."""


def _require(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"{what}: missing columns {missing}")


def _raise(what: str, problems: list[str]) -> None:
    if problems:
        extra = f" (+{len(problems) - MAX_REPORTED} more)" if len(problems) > MAX_REPORTED else ""
        raise ContractError(f"{what}: " + "; ".join(problems[:MAX_REPORTED]) + extra)


def _text(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), "").astype(str).str.strip()


def _times(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse ISO-8601 timestamps to naive UTC. Returns (parsed, unparseable_mask); empty values are NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series.dt.tz_convert(None) if series.dt.tz is not None else series
        return parsed, pd.Series(False, index=series.index)
    text = _text(series)
    parsed = pd.to_datetime(text.where(text != ""), errors="coerce", utc=True, format="ISO8601").dt.tz_localize(None)
    return parsed, (text != "") & parsed.isna()


def _days(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(_text(series), errors="coerce", format="ISO8601")
    return parsed.dt.date, parsed.isna()


def _bools(series: pd.Series) -> pd.Series:
    lookup = {"true": True, "1": True, "false": False, "0": False}
    return _text(series).str.lower().map(lambda v: lookup.get(v))


def _split(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def validate_window_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly model output: one row per (entity, window_start); anomaly_score in [0, 1], higher = more anomalous."""
    _require(df, SCORE_COLUMNS, "window scores")
    out, problems = df.copy(), []
    score = pd.to_numeric(out["anomaly_score"], errors="coerce")
    if score.isna().any() or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        problems.append("anomaly_score must be finite and within [0, 1]")
    out["window_start"], bad = _times(out["window_start"])
    if bad.any() or out["window_start"].isna().any():
        problems.append("window_start must be a timestamp")
    if out.duplicated(["entity", "window_start"]).any():
        problems.append("duplicate (entity, window_start) rows")
    _raise("window scores", problems)
    out["anomaly_score"] = score
    return out


def check_score_direction(score: Callable[[pd.DataFrame], np.ndarray],
                          typical: pd.DataFrame, atypical: pd.DataFrame) -> tuple[float, float]:
    """Contract check for an anomaly scorer: clearly unusual windows must score higher than benign-like ones.

    `score` maps a frame with MODEL_INPUT_COLUMNS to scores in [0, 1]. Returns (mean typical, mean atypical).
    """
    t = np.asarray(score(typical), dtype=float)
    a = np.asarray(score(atypical), dtype=float)
    problems = []
    for name, s, frame in (("typical", t, typical), ("atypical", a, atypical)):
        if s.shape != (len(frame),):
            problems.append(f"{name}: expected {len(frame)} scores, got shape {s.shape}")
        elif not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
            problems.append(f"{name}: scores must be finite and within [0, 1]")
    if not problems and a.mean() <= t.mean():
        problems.append(f"atypical windows scored {a.mean():.3f} on average vs {t.mean():.3f} for typical ones; "
                        "higher must mean more anomalous")
    _raise("anomaly score direction", problems)
    return float(t.mean()), float(a.mean())


def validate_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """Explanation layer output: per-feature deviation of a window from the benign training baseline."""
    _require(df, EXPLANATION_COLUMNS, "explanations")
    out, problems = df.copy(), []
    unknown = sorted(set(out["feature"]) - set(FEATURES))
    if unknown:
        problems.append(f"unknown features {unknown}")
    for col in ("value", "baseline_median", "baseline_scale", "deviation"):
        values = pd.to_numeric(out[col], errors="coerce")
        if not np.isfinite(values).all():
            problems.append(f"{col} must be finite")
        out[col] = values
    if (out["baseline_scale"] <= 0).any():
        problems.append("baseline_scale must be positive")
    out["window_start"], bad = _times(out["window_start"])
    if bad.any() or out["window_start"].isna().any():
        problems.append("window_start must be a timestamp")
    _raise("explanations", problems)
    return out


def validate_stage_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Stage inference output: one row per observed stage signal, pointing back to an event id."""
    _require(df, STAGE_SIGNAL_COLUMNS, "stage signals")
    out, problems = df.copy(), []
    unknown = sorted(set(_text(out["stage"])) - set(STAGES))
    if unknown:
        problems.append(f"unknown stages {unknown}")
    out["ts"], bad = _times(out["ts"])
    if bad.any() or out["ts"].isna().any():
        problems.append("ts must be a timestamp")
    eid = pd.to_numeric(out["eid"], errors="coerce")
    if eid.isna().any() or (eid < 0).any() or (eid % 1 != 0).any():
        problems.append("eid must be a non-negative integer event id")
    if (_text(out["evidence"]) == "").any():
        problems.append("evidence must not be empty")
    _raise("stage signals", problems)
    out["eid"] = eid.astype(int)
    return out


def validate_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Progression output: one row per (entity, day).

    stages_detected / stage_times are pipe-separated and aligned, listed in chronological order (repeats allowed).
    current_stage is the most advanced detected stage in STAGES order; next_expected_stage is the stage after it
    ("" when nothing was detected or current_stage is the last stage). stages_completed counts distinct stages.
    """
    _require(df, PROGRESSION_COLUMNS, "progression")
    out, problems = df.copy(), []
    score = pd.to_numeric(out["progression_score"], errors="coerce")
    if score.isna().any() or ((score < 0) | (score > 100)).any():
        problems.append("progression_score must be within [0, 100]")
    out["first_warning_ts"], bad = _times(out["first_warning_ts"])
    if bad.any():
        problems.append("first_warning_ts must be ISO-8601 or empty")
    out["day"], bad = _days(out["day"])
    if bad.any():
        problems.append("day must be YYYY-MM-DD")
    if out.duplicated(["entity", "day"]).any():
        problems.append("duplicate (entity, day) rows")

    for i, row in enumerate(df.itertuples(index=False)):
        where = f"row {i}"
        stages, raw_times = _split(row.stages_detected), _split(row.stage_times)
        unknown = [s for s in stages if s not in STAGE_INDEX]
        if unknown:
            problems.append(f"{where}: unknown stages {unknown}")
            continue
        if len(raw_times) != len(stages):
            problems.append(f"{where}: stage_times must align with stages_detected")
            continue
        times = pd.to_datetime(raw_times, errors="coerce", utc=True, format="ISO8601")
        if times.isna().any():
            problems.append(f"{where}: unparseable stage_times")
            continue
        if (times[1:] < times[:-1]).any():
            problems.append(f"{where}: stages_detected must be listed in chronological order")
        if pd.to_numeric(row.stages_completed, errors="coerce") != len(set(stages)):
            problems.append(f"{where}: stages_completed must equal the number of distinct detected stages")
        current = max(stages, key=STAGE_INDEX.get) if stages else ""
        nxt = STAGES[STAGE_INDEX[current] + 1] if current and current != STAGES[-1] else ""
        if _split(row.current_stage)[:1] != ([current] if current else []):
            problems.append(f"{where}: current_stage must be {current or 'empty'}")
        if _split(row.next_expected_stage)[:1] != ([nxt] if nxt else []):
            problems.append(f"{where}: next_expected_stage must be {nxt or 'empty'}")
    _raise("progression", problems)
    out["progression_score"] = score
    return out


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Final per-(split, detector, entity, day) predictions consumed by the evaluation harness."""
    _require(df, PREDICTION_COLUMNS, "predictions")
    out, problems = df.copy(), []
    for col in ("split", "detector", "entity", "predicted_class", "current_stage", "explanation"):
        out[col] = _text(out[col])
    if bad := sorted(set(out["split"]) - set(SPLITS)):
        problems.append(f"unknown splits {bad}")
    if bad := sorted(set(out["detector"]) - set(DETECTORS)):
        problems.append(f"unknown detectors {bad}")
    if bad := sorted(set(out["predicted_class"]) - set(CLASSES)):
        problems.append(f"unknown predicted_class values {bad}")
    if bad := sorted(set(out["current_stage"]) - set(STAGES) - {""}):
        problems.append(f"unknown current_stage values {bad}")
    if (out["entity"] == "").any():
        problems.append("entity must not be empty")

    alert = _bools(out["alert"])
    if alert.isna().any():
        problems.append("alert must be boolean")
    else:
        out["alert"] = alert.astype(bool)
        if (out["alert"] != (out["predicted_class"] != "Normal")).any():
            problems.append("alert must equal (predicted_class != 'Normal')")
    risk = pd.to_numeric(out["risk_score"], errors="coerce")
    if risk.isna().any() or ((risk < 0) | (risk > 100)).any():
        problems.append("risk_score must be within [0, 100]")
    completed = pd.to_numeric(out["stages_completed"], errors="coerce")
    if completed.isna().any() or (completed % 1 != 0).any() or ((completed < 0) | (completed > len(STAGES))).any():
        problems.append(f"stages_completed must be an integer in 0..{len(STAGES)}")
    out["day"], bad = _days(out["day"])
    if bad.any():
        problems.append("day must be YYYY-MM-DD")
    out["first_warning_ts"], bad = _times(out["first_warning_ts"])
    if bad.any():
        problems.append("first_warning_ts must be ISO-8601 or empty")
    elif alert.notna().all() and (out["alert"] & out["first_warning_ts"].isna()).any():
        problems.append("every alert needs a first_warning_ts")
    if out.duplicated(["split", "detector", "entity", "day"]).any():
        problems.append("duplicate (split, detector, entity, day) rows")
    _raise("predictions", problems)
    out["risk_score"], out["stages_completed"] = risk, completed.astype(int)
    return out
