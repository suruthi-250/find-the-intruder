"""Thin prediction layer combining anomaly scores and behavioral progression.

This module does not train a model, inspect labels, or perform response actions.
It converts observable anomaly/rule evidence into the project's prediction contract.
"""

from __future__ import annotations

import math

import pandas as pd

from .contracts import CLASSES, DETECTORS, PREDICTION_COLUMNS, STAGES, validate_predictions


ANOMALY_NORMAL_MAX = 0.50
ANOMALY_SUSPICIOUS_MAX = 0.75
STAGE_SUSPICIOUS_MIN = 1
STAGE_FAILED_MIN = 2
PROGRESSION_INTRUSION_MIN = 50.0
PROGRESSION_HIGH_MIN = 75.0


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _as_frame(value: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return value.copy()


def _normalise_anomaly_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"entity", "window_start", "anomaly_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"anomaly_scores missing required columns: {missing}")

    out = scores.copy()
    out["window_start"] = pd.to_datetime(out["window_start"], errors="coerce")
    if out["window_start"].isna().any():
        raise ValueError("anomaly_scores contains invalid window_start values")
    out["anomaly_score"] = pd.to_numeric(out["anomaly_score"], errors="coerce")
    if out["anomaly_score"].isna().any():
        raise ValueError("anomaly_scores contains invalid anomaly scores")
    if ((out["anomaly_score"] < 0) | (out["anomaly_score"] > 1)).any():
        raise ValueError("anomaly_score must be within [0, 1]")

    out["entity"] = out["entity"].astype(str)
    out["day"] = out["window_start"].dt.date.astype(str)

    # Window-level evidence is kept so warning times can point at the window
    # that actually triggered an alert.
    return (
        out[["entity", "day", "window_start", "anomaly_score"]]
        .sort_values(["entity", "day", "window_start"], kind="stable")
        .reset_index(drop=True)
    )


def _anomaly_by_day(windows: pd.DataFrame) -> pd.DataFrame:
    # Evaluation is entity-day based. The strongest observed window is the
    # conservative day-level anomaly evidence.
    return windows.groupby(["entity", "day"], as_index=False)["anomaly_score"].max()


def _normalise_stage_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Unordered stage evidence per entity-day, taken from the raw stage signals.

    This is B's input: the set of distinct stages observed (shown in canonical
    order) and the time of the first signal. Order and timing are ignored.
    """
    columns = ["entity", "day", "raw_stages", "raw_first_ts"]
    if signals.empty:
        return pd.DataFrame(columns=columns)

    missing = sorted({"entity", "ts", "stage"} - set(signals.columns))
    if missing:
        raise ValueError(f"stage_signals missing required columns: {missing}")

    out = signals.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    if out["ts"].isna().any():
        raise ValueError("stage_signals contains invalid timestamps")
    out = out.loc[out["stage"].isin(STAGES)].copy()
    out["entity"] = out["entity"].astype(str)
    out["day"] = out["ts"].dt.date.astype(str)

    rows = []
    for (entity, day), group in out.groupby(["entity", "day"], sort=True):
        distinct = sorted(set(group["stage"]), key=STAGES.index)
        rows.append({
            "entity": entity,
            "day": day,
            "raw_stages": "|".join(distinct),
            "raw_first_ts": group["ts"].min(),
        })
    return pd.DataFrame(rows, columns=columns)


def _normalise_progression(progressions: pd.DataFrame) -> pd.DataFrame:
    if progressions.empty:
        return pd.DataFrame(
            columns=[
                "entity", "day", "stages_detected", "stages_completed",
                "current_stage", "progression_score", "first_warning_ts"
            ]
        )

    required = {
        "entity", "day", "stages_detected", "stages_completed",
        "current_stage", "progression_score", "first_warning_ts",
    }
    missing = sorted(required - set(progressions.columns))
    if missing:
        raise ValueError(f"progressions missing required columns: {missing}")

    out = progressions.copy()
    out["entity"] = out["entity"].astype(str)
    out["day"] = pd.to_datetime(out["day"], errors="coerce").dt.date.astype(str)
    if out["day"].isna().any():
        raise ValueError("progressions contains invalid days")

    out["stages_completed"] = pd.to_numeric(
        out["stages_completed"], errors="raise"
    ).astype(int)
    out["progression_score"] = pd.to_numeric(
        out["progression_score"], errors="raise"
    ).astype(float)

    if ((out["stages_completed"] < 0) | (out["stages_completed"] > 6)).any():
        raise ValueError("stages_completed must be within [0, 6]")
    if ((out["progression_score"] < 0) | (out["progression_score"] > 100)).any():
        raise ValueError("progression_score must be within [0, 100]")

    return out


def _stage_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x for x in text.split("|") if x in STAGES]


def _risk_anomaly(score: float) -> float:
    # Map [0,1] anomaly evidence to [0,100] without changing its direction.
    return round(float(score) * 100.0, 2)


def _risk_stage_count(anomaly: float, stages_completed: int) -> float:
    # Both terms on the 0-100 scale: anomaly in [0,1] is scaled before mixing.
    stage_component = min(max(stages_completed, 0), 6) / 6.0 * 100.0
    return round(0.55 * float(anomaly) * 100.0 + 0.45 * stage_component, 2)


def _risk_progression(anomaly: float, progression_score: float) -> float:
    # Both terms on the 0-100 scale: anomaly in [0,1] is scaled before mixing.
    return round(0.45 * float(anomaly) * 100.0 + 0.55 * progression_score, 2)


def _risk(detector: str, anomaly: float, stages_completed: int, progression_score: float) -> float:
    if detector == "A_anomaly_only":
        return _risk_anomaly(anomaly)
    if detector == "B_stage_count":
        return _risk_stage_count(anomaly, stages_completed)
    return _risk_progression(anomaly, progression_score)


def _classification(
    detector: str,
    risk: float,
    anomaly: float,
    stages: list[str],
    progression_score: float,
) -> str:
    if detector == "A_anomaly_only":
        if risk < ANOMALY_NORMAL_MAX * 100:
            return "Normal"
        return "Suspicious"

    # Stage evidence is counted by distinct stage, never by repeated signals.
    distinct = list(dict.fromkeys(stages))

    advanced = {
        "COMPROMISE", "PRIV_ESC", "PERSISTENCE", "EXFIL"
    }
    if detector == "C_progression" and progression_score >= PROGRESSION_INTRUSION_MIN:
        return "Intrusion"
    if advanced.intersection(distinct) and risk >= ANOMALY_NORMAL_MAX * 100:
        return "Intrusion"

    if (
        len(distinct) >= STAGE_FAILED_MIN
        or ("RECON" in distinct and "CRED_ATTACK" in distinct)
    ):
        return "Failed attack"

    if risk < ANOMALY_NORMAL_MAX * 100 and len(distinct) == 0:
        return "Normal"

    if risk >= ANOMALY_NORMAL_MAX * 100 or len(distinct) >= STAGE_SUSPICIOUS_MIN:
        return "Suspicious"

    return "Normal"


def _explanation(
    detector: str,
    anomaly: float,
    stages: list[str],
    progression_score: float,
    predicted_class: str,
) -> str:
    parts: list[str] = []

    if anomaly >= ANOMALY_NORMAL_MAX:
        parts.append(f"anomaly evidence {anomaly:.2f}")
    if stages:
        parts.append("stages: " + ", ".join(dict.fromkeys(stages)))
    if detector == "C_progression" and progression_score > 0:
        parts.append(f"progression {progression_score:.1f}/100")
    if not parts:
        parts.append("no strong behavioral evidence")

    return f"{predicted_class}: " + "; ".join(parts)


def _present(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _warning_ts(
    detector: str,
    entity: str,
    day: str,
    stage_warning: object,
    windows: pd.DataFrame | None,
) -> str:
    """Time of the evidence that made this detector alert.

    One stage of evidence is enough on its own for B and C to alert, so their
    stage evidence time (B: first raw signal, C: progression first_warning_ts)
    is a candidate. The other candidate is the first anomaly window whose score
    alone makes this detector alert. The earlier of the two is returned; A only
    has the anomaly candidate. An alert with neither is a logic error: no time
    is invented.
    """
    candidates = []
    if _present(stage_warning):
        candidates.append(pd.Timestamp(stage_warning))

    if windows is not None:
        for window in windows.itertuples(index=False):
            score = float(window.anomaly_score)
            risk = _risk(detector, score, 0, 0.0)
            if _classification(detector, risk, score, [], 0.0) != "Normal":
                candidates.append(pd.Timestamp(window.window_start))
                break

    if not candidates:
        raise RuntimeError(
            f"{detector} alerted on {entity} {day} without timestamped evidence"
        )
    return min(candidates).isoformat()


def build_predictions(
    anomaly_scores: pd.DataFrame,
    progressions: pd.DataFrame | None = None,
    *,
    split: str,
    detector: str,
    stage_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build validated entity-day predictions for one detector and split.

    A uses anomaly scores only. B uses anomaly scores plus the unordered stage
    evidence in `stage_signals` (raw signals; without them B has no stage
    evidence). C uses anomaly scores plus the progression output.
    """
    if split not in {"train", "val", "test_id", "test_ood"}:
        raise ValueError(f"unknown split: {split}")
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector: {detector}")

    windows = _normalise_anomaly_scores(_as_frame(anomaly_scores, "anomaly_scores"))
    anomaly = _anomaly_by_day(windows)
    progression = _normalise_progression(
        _as_frame(progressions, "progressions")
    )
    raw = _normalise_stage_signals(_as_frame(stage_signals, "stage_signals"))

    keys = ["entity", "day"]
    frames = [frame for frame in (anomaly, progression, raw) if not frame.empty]
    if not frames:
        return _empty_predictions()

    base = frames[0]
    for frame in frames[1:]:
        base = base.merge(frame, on=keys, how="outer")
    base = base.sort_values(keys, kind="stable").reset_index(drop=True)

    for column, default in {
        "anomaly_score": 0.0,
        "stages_detected": "",
        "stages_completed": 0,
        "current_stage": "",
        "progression_score": 0.0,
        "first_warning_ts": "",
        "raw_stages": "",
    }.items():
        if column not in base.columns:
            base[column] = default
        else:
            base[column] = base[column].fillna(default)
    if "raw_first_ts" not in base.columns:
        base["raw_first_ts"] = pd.NaT

    windows_by_day = {
        key: group for key, group in windows.groupby(keys, sort=False)
    }

    rows = []
    for _, row in base.iterrows():
        entity, day = str(row["entity"]), str(row["day"])
        anomaly_score = float(row["anomaly_score"])
        progression_score = float(row["progression_score"])

        if detector == "A_anomaly_only":
            stages: list[str] = []
            completed, current_stage, stage_warning = 0, "", None
        elif detector == "B_stage_count":
            stages = _stage_list(row["raw_stages"])
            completed = len(stages)
            current_stage = max(stages, key=STAGES.index) if stages else ""
            stage_warning = row["raw_first_ts"] if stages else None
        else:
            stages = _stage_list(row["stages_detected"])
            completed = int(row["stages_completed"])
            current_stage = str(row["current_stage"] or "")
            stage_warning = row["first_warning_ts"] if stages else None

        risk = _risk(detector, anomaly_score, completed, progression_score)
        predicted = _classification(
            detector, risk, anomaly_score, stages, progression_score
        )
        alert = predicted != "Normal"

        warning = ""
        if alert:
            warning = _warning_ts(
                detector, entity, day, stage_warning,
                windows_by_day.get((entity, day)),
            )

        rows.append({
            "split": split,
            "detector": detector,
            "entity": str(row["entity"]),
            "day": str(row["day"]),
            "risk_score": float(risk),
            "predicted_class": predicted,
            "alert": bool(alert),
            "first_warning_ts": warning,
            "current_stage": current_stage,
            "stages_completed": completed,
            "explanation": _explanation(
                detector, anomaly_score, stages, progression_score, predicted
            ),
        })

    result = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    return validate_predictions(result)


def build_all_detectors(
    anomaly_scores: pd.DataFrame,
    progressions: pd.DataFrame | None,
    *,
    split: str,
    stage_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build A/B/C predictions using the same entity-day population."""
    frames = [
        build_predictions(
            anomaly_scores,
            progressions,
            split=split,
            detector=detector,
            stage_signals=stage_signals,
        )
        for detector in DETECTORS
    ]
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return _empty_predictions()
    return pd.concat(nonempty, ignore_index=True)


__all__ = ["build_predictions", "build_all_detectors"]
