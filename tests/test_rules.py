from __future__ import annotations

import pandas as pd
import pytest

from intruder.config import Config
from intruder.contracts import ContractError, validate_progression, validate_stage_signals
from intruder.rules import detect_stage_signals, evaluate_progression


def event(
    eid: int,
    ts: str,
    source_ip: str = "10.0.0.50",
    event_type: str = "CONNECTION",
    status: str = "DENY",
    *,
    destination_ip: str | None = "10.0.0.20",
    host: str | None = "app-01",
    username: str | None = "alice",
    port: int | None = 22,
    bytes_: int = 0,
    privilege: str | None = None,
) -> dict:
    return {
        "eid": eid,
        "ts": pd.Timestamp(ts),
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "host": host,
        "username": username,
        "event_type": event_type,
        "status": status,
        "port": port,
        "bytes": bytes_,
        "http_status": None,
        "request_marker": "NONE",
        "privilege": privilege,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def stages(result: pd.DataFrame) -> list[str]:
    return result["stage"].tolist()


def test_repeated_multi_port_connections_produce_recon():
    rows = [
        event(i, f"2026-10-04 10:00:{i:02d}", port=20 + i)
        for i in range(4)
    ]
    result = detect_stage_signals(frame(rows))
    assert "RECON" in stages(result)


def test_single_connection_does_not_produce_recon():
    result = detect_stage_signals(frame([event(0, "2026-10-04 10:00:00")]))
    assert result.empty


def test_repeated_failed_auth_produces_cred_attack():
    rows = [
        event(i, f"2026-10-04 10:00:{i:02d}", event_type="FAILED_LOGIN",
              status="FAILURE", port=None)
        for i in range(3)
    ]
    result = detect_stage_signals(frame(rows))
    assert "CRED_ATTACK" in stages(result)


def test_single_failed_login_is_not_cred_attack():
    row = event(
        0, "2026-10-04 10:00:00",
        event_type="FAILED_LOGIN", status="FAILURE", port=None
    )
    assert detect_stage_signals(frame([row])).empty


def test_suspicious_auth_followed_by_success_produces_compromise():
    rows = [
        event(0, "2026-10-04 10:00:00", event_type="FAILED_LOGIN",
              status="FAILURE", port=None),
        event(1, "2026-10-04 10:01:00", event_type="FAILED_LOGIN",
              status="FAILURE", port=None),
        event(2, "2026-10-04 10:02:00", event_type="SUCCESS_LOGIN",
              status="SUCCESS", port=None),
    ]
    result = detect_stage_signals(frame(rows))
    assert "COMPROMISE" in stages(result)


def test_ordinary_success_does_not_produce_compromise():
    row = event(
        0, "2026-10-04 10:00:00",
        event_type="SUCCESS_LOGIN", status="SUCCESS", port=None
    )
    assert detect_stage_signals(frame([row])).empty


def test_elevated_shell_produces_priv_esc():
    row = event(
        0, "2026-10-04 10:00:00",
        event_type="SUDO_EVENT", status="SUCCESS", source_ip=None,
        port=None, privilege="ELEVATED_SHELL"
    )
    result = detect_stage_signals(frame([row]))
    assert "PRIV_ESC" in stages(result)
    assert result.iloc[0]["eid"] == 0


def test_routine_sudo_is_not_strong_priv_esc_signal():
    row = event(
        0, "2026-10-04 10:00:00",
        event_type="SUDO_EVENT", status="SUCCESS", source_ip=None,
        port=None, privilege="ROUTINE"
    )
    result = detect_stage_signals(frame([row]))
    assert "PRIV_ESC" not in stages(result)


def test_new_account_produces_persistence():
    row = event(
        0, "2026-10-04 10:00:00",
        event_type="NEW_ACCOUNT", status="SUCCESS", source_ip=None,
        port=None, username="newuser"
    )
    result = detect_stage_signals(frame([row]))
    assert "PERSISTENCE" in stages(result)


def test_large_external_transfer_after_compromise_produces_exfil():
    rows = [
        event(0, "2026-10-04 10:00:00", event_type="FAILED_LOGIN",
              status="FAILURE", port=None),
        event(1, "2026-10-04 10:01:00", event_type="FAILED_LOGIN",
              status="FAILURE", port=None),
        event(2, "2026-10-04 10:02:00", event_type="SUCCESS_LOGIN",
              status="SUCCESS", port=None),
        event(3, "2026-10-04 10:20:00", event_type="EXTERNAL_TRANSFER",
              status="SUCCESS", destination_ip="192.0.2.10",
              port=None, bytes_=1_000_000),
    ]
    result = detect_stage_signals(frame(rows))
    assert "EXFIL" in stages(result)


def test_small_external_transfer_does_not_produce_exfil():
    rows = [
        event(0, "2026-10-04 10:00:00", event_type="FAILED_LOGIN",
              status="FAILURE", port=None),
        event(1, "2026-10-04 10:01:00", event_type="FAILED_LOGIN",
              status="FAILURE", port=None),
        event(2, "2026-10-04 10:02:00", event_type="SUCCESS_LOGIN",
              status="SUCCESS", port=None),
        event(3, "2026-10-04 10:20:00", event_type="EXTERNAL_TRANSFER",
              status="SUCCESS", destination_ip="192.0.2.10",
              port=None, bytes_=20_000),
    ]
    result = detect_stage_signals(frame(rows))
    assert "EXFIL" not in stages(result)


def signal(entity, ts, stage, eid):
    return {
        "entity": entity,
        "ts": pd.Timestamp(ts),
        "stage": stage,
        "eid": eid,
        "evidence": f"{stage} evidence",
    }


def test_repeated_stage_counts_once():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
        signal("10.0.0.50", "2026-10-04 10:01:00", "RECON", 1),
        signal("10.0.0.50", "2026-10-04 10:02:00", "CRED_ATTACK", 2),
    ])
    result = evaluate_progression(signals)
    row = result.iloc[0]
    assert row["stages_completed"] == 2
    assert row["current_stage"] == "CRED_ATTACK"
    assert row["next_expected_stage"] == "COMPROMISE"


def test_missing_intermediate_stages_are_allowed():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
        signal("10.0.0.50", "2026-10-04 10:02:00", "PRIV_ESC", 1),
        signal("10.0.0.50", "2026-10-04 10:04:00", "EXFIL", 2),
    ])
    result = evaluate_progression(signals)
    row = result.iloc[0]
    assert row["stages_completed"] == 3
    assert row["current_stage"] == "EXFIL"
    assert row["next_expected_stage"] == ""


def test_correct_chronological_progression_is_recognized():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
        signal("10.0.0.50", "2026-10-04 10:03:00", "CRED_ATTACK", 1),
        signal("10.0.0.50", "2026-10-04 10:05:00", "COMPROMISE", 2),
        signal("10.0.0.50", "2026-10-04 10:08:00", "PRIV_ESC", 3),
        signal("10.0.0.50", "2026-10-04 10:10:00", "PERSISTENCE", 4),
        signal("10.0.0.50", "2026-10-04 10:12:00", "EXFIL", 5),
    ])
    result = evaluate_progression(signals)
    row = result.iloc[0]
    assert row["stages_completed"] == 6
    assert row["progression_score"] == 100.0
    assert row["first_warning_ts"].startswith("2026-10-04T10:00:00")


def test_out_of_order_progression_is_not_treated_as_complete():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
        signal("10.0.0.50", "2026-10-04 10:03:00", "PRIV_ESC", 1),
        signal("10.0.0.50", "2026-10-04 10:04:00", "CRED_ATTACK", 2),
        signal("10.0.0.50", "2026-10-04 10:05:00", "COMPROMISE", 3),
    ])
    result = evaluate_progression(signals)
    row = result.iloc[0]
    assert row["stages_completed"] < 6
    assert row["stages_detected"] == "RECON|PRIV_ESC"


def test_first_warning_is_earliest_qualifying_signal():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
        signal("10.0.0.50", "2026-10-04 10:03:00", "CRED_ATTACK", 1),
    ])
    result = evaluate_progression(signals)
    assert result.iloc[0]["first_warning_ts"].startswith("2026-10-04T10:00:00")


def test_stage_signals_satisfy_contract():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
    ])
    checked = validate_stage_signals(detect_stage_signals(
        frame([
            event(0, "2026-10-04 10:00:00", port=20),
            event(1, "2026-10-04 10:00:01", port=21),
            event(2, "2026-10-04 10:00:02", port=22),
            event(3, "2026-10-04 10:00:03", port=23),
        ])
    ))
    assert list(checked.columns) == [
        "entity", "ts", "stage", "eid", "evidence"
    ]


def test_progression_output_satisfies_contract():
    signals = pd.DataFrame([
        signal("10.0.0.50", "2026-10-04 10:00:00", "RECON", 0),
        signal("10.0.0.50", "2026-10-04 10:02:00", "CRED_ATTACK", 1),
    ])
    checked = validate_progression(evaluate_progression(signals))
    assert checked.iloc[0]["stages_completed"] == 2


def test_empty_progression_is_valid():
    checked = validate_progression(
        evaluate_progression(pd.DataFrame(columns=[
            "entity", "ts", "stage", "eid", "evidence"
        ]))
    )
    assert checked.empty
