"""Regression tests for the Phase 7.2 prediction correctness fixes."""
import pandas as pd
import pytest

from intruder.prediction import _warning_ts, build_all_detectors, build_predictions

DAY = "2026-10-04"


def windows(*rows):
    return pd.DataFrame([{"entity": e, "window_start": f"{DAY} {t}", "anomaly_score": s} for e, t, s in rows])


def signals(*rows):
    return pd.DataFrame([{"entity": e, "ts": f"{DAY} {t}", "stage": st, "eid": i, "evidence": "observed"}
                         for i, (e, t, st) in enumerate(rows)])


def progression(entity, stages, times, score):
    distinct = list(dict.fromkeys(stages))
    order = ["RECON", "CRED_ATTACK", "COMPROMISE", "PRIV_ESC", "PERSISTENCE", "EXFIL"]
    current = max(distinct, key=order.index) if distinct else ""
    nxt = order[order.index(current) + 1] if current and current != "EXFIL" else ""
    return pd.DataFrame([{
        "entity": entity, "day": DAY, "stages_detected": "|".join(stages),
        "stage_times": "|".join(f"{DAY}T{t}" for t in times), "stages_completed": len(distinct),
        "current_stage": current, "next_expected_stage": nxt, "progression_score": score,
        "first_warning_ts": f"{DAY}T{times[0]}" if times else "",
    }])


def one(df, detector):
    return df[df.detector == detector].iloc[0]


def test_anomaly_only_warns_at_its_own_first_triggering_window():
    w = windows(("10.0.0.5", "09:00:00", 0.30), ("10.0.0.5", "10:00:00", 0.70), ("10.0.0.5", "11:00:00", 0.95))
    p = progression("10.0.0.5", ["RECON"], ["08:00:00"], 16.67)    # earlier stage evidence must be ignored by A
    row = build_predictions(w, p, split="val", detector="A_anomaly_only").iloc[0]
    assert row.alert
    assert row.first_warning_ts == pd.Timestamp(f"{DAY} 10:00:00")


def test_anomaly_driven_alert_uses_the_triggering_window_not_midnight():
    # No stages at all; B alerts on anomaly alone only once 0.55 * score * 100 reaches 50.
    w = windows(("10.0.0.6", "13:00:00", 0.60), ("10.0.0.6", "14:00:00", 0.95))
    out = build_all_detectors(w, None, split="val")
    b = one(out, "B_stage_count")
    assert b.predicted_class == "Suspicious"
    assert b.first_warning_ts == pd.Timestamp(f"{DAY} 14:00:00")
    assert one(out, "A_anomaly_only").first_warning_ts == pd.Timestamp(f"{DAY} 13:00:00")
    alerts = out[out.alert]
    assert not (alerts.first_warning_ts == pd.Timestamp(DAY)).any()


def test_every_alert_time_is_an_evidence_time():
    w = windows(("10.0.0.7", "09:30:00", 0.20), ("10.0.0.8", "16:10:00", 0.97))
    s = signals(("10.0.0.7", "12:00:00", "RECON"), ("10.0.0.7", "12:01:00", "CRED_ATTACK"))
    p = progression("10.0.0.7", ["RECON", "CRED_ATTACK"], ["12:00:00", "12:01:00"], 33.33)
    out = build_all_detectors(w, p, split="val", stage_signals=s)
    evidence = set(pd.to_datetime(w.window_start)) | set(pd.to_datetime(s.ts))
    assert out.alert.any()
    assert set(out.loc[out.alert, "first_warning_ts"]) <= evidence


def test_alert_without_evidence_is_a_logic_error():
    with pytest.raises(RuntimeError, match="without timestamped evidence"):
        _warning_ts("C_progression", "10.0.0.9", DAY, None, None)


def test_anomaly_is_scaled_to_0_100_before_combining():
    w = windows(("10.0.0.10", "10:00:00", 0.80))
    out = build_all_detectors(w, None, split="val")
    assert one(out, "A_anomaly_only").risk_score == 80.0
    assert one(out, "B_stage_count").risk_score == 44.0          # 0.55 * 80
    assert one(out, "C_progression").risk_score == 36.0          # 0.45 * 80

    s = signals(("10.0.0.10", "10:01:00", "RECON"), ("10.0.0.10", "10:02:00", "CRED_ATTACK"))
    p = progression("10.0.0.10", ["RECON", "CRED_ATTACK"], ["10:01:00", "10:02:00"], 50.0)
    out = build_all_detectors(w, p, split="val", stage_signals=s)
    assert one(out, "B_stage_count").risk_score == 59.0          # 0.55 * 80 + 0.45 * (2/6 * 100)
    assert one(out, "C_progression").risk_score == 63.5          # 0.45 * 80 + 0.55 * 50


def test_b_uses_raw_unordered_signals_not_the_truncated_progression():
    w = windows(("10.0.0.11", "10:00:00", 0.20))
    s = signals(("10.0.0.11", "10:00:00", "PRIV_ESC"), ("10.0.0.11", "10:05:00", "RECON"),
                ("10.0.0.11", "10:06:00", "CRED_ATTACK"))
    p = progression("10.0.0.11", ["PRIV_ESC"], ["10:00:00"], 16.67)   # order filter kept only the prefix
    out = build_all_detectors(w, p, split="val", stage_signals=s)
    b, c = one(out, "B_stage_count"), one(out, "C_progression")
    assert (b.stages_completed, b.current_stage) == (3, "PRIV_ESC")
    assert (c.stages_completed, c.current_stage) == (1, "PRIV_ESC")
    assert b.predicted_class == "Failed attack" and c.predicted_class == "Suspicious"
    assert b.first_warning_ts == pd.Timestamp(f"{DAY} 10:00:00")

    without = build_predictions(w, p, split="val", detector="B_stage_count").iloc[0]
    assert without.stages_completed == 0                         # B never reads progression stages


def test_repeated_signals_count_as_one_stage():
    w = windows(("10.0.0.12", "10:00:00", 0.10))
    s = signals(*[("10.0.0.12", f"10:0{i}:00", "RECON") for i in range(5)])
    p = progression("10.0.0.12", ["RECON"] * 3, ["10:00:00", "10:01:00", "10:02:00"], 16.67)
    out = build_all_detectors(w, p, split="val", stage_signals=s)
    for detector in ("B_stage_count", "C_progression"):
        row = one(out, detector)
        assert row.stages_completed == 1
        assert row.predicted_class == "Suspicious"               # one distinct stage is not a failed attack
        assert row.explanation.count("RECON") == 1
