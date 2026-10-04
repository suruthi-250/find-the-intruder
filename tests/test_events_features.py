import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intruder.config import Config
from intruder.events import LabelLeakError, load_events, load_hosts
from intruder.features import FEATURES, build_features, model_matrix

HEADER = "timestamp,source_ip,destination_ip,host,username,event_type,status,port,bytes,http_status,request_marker,privilege"


def csv(*rows: str) -> io.StringIO:
    return io.StringIO("\n".join([HEADER, *rows]) + "\n")


def features(*rows: str, cfg: Config | None = None) -> pd.DataFrame:
    events, report = load_events(csv(*rows))
    assert report.skipped == 0, report.reasons
    return build_features(events, cfg or Config()).set_index(["entity", "window_start"])


# --- loading / validation ---------------------------------------------------------------------

def test_valid_rows_are_loaded_sorted_and_numbered():
    events, report = load_events(csv(
        "2026-10-04T10:01:00Z,10.0.1.5,10.0.2.1,app01,ana,SUCCESS_LOGIN,SUCCESS,22,,,,",
        "2026-10-04T10:00:00Z,10.0.1.5,10.0.2.1,app01,ana,FAILED_LOGIN,FAILURE,22,,,,",
    ))
    assert (report.rows, report.valid, report.skipped) == (2, 2, 0)
    assert list(events.event_type) == ["FAILED_LOGIN", "SUCCESS_LOGIN"]
    assert list(events.eid) == [0, 1]
    assert events.request_marker.eq("NONE").all()


def test_malformed_rows_are_skipped_and_counted_by_reason():
    events, report = load_events(csv(
        "not-a-time,10.0.1.5,,,ana,FAILED_LOGIN,FAILURE,22,,,,",
        "2026-10-04T10:00:00Z,10.0.1.5,,,ana,TELEPORT,FAILURE,22,,,,",
        "2026-10-04T10:00:00Z,10.0.1.5,10.0.2.1,,,CONNECTION,ALLOW,99999,,,,",
        "2026-10-04T10:00:00Z,8.8.8.8,10.0.2.1,,,CONNECTION,ALLOW,443,,,,",       # non-synthetic external
        "2026-10-04T10:00:00Z,10.0.1.5,10.0.2.1,,,CONNECTION,ALLOW,443,-5,,,",
        "2026-10-04T10:00:00Z,,10.0.2.1,,,CONNECTION,ALLOW,443,,,,",
        "2026-10-04T10:00:00Z,10.0.1.5,,app01,ana,SUDO_EVENT,SUCCESS,,,,,ROOTISH",
        "2026-10-04T10:00:00Z,198.51.100.7,10.0.2.1,,,WEB_REQUEST,SUCCESS,443,100,200,NONE,",  # valid
    ))
    assert report.valid == 1 and report.skipped == 7
    assert report.reasons == {
        "bad timestamp": 1, "unknown event_type": 1, "bad port": 1, "invalid or non-synthetic source_ip": 1,
        "bad bytes": 1, "missing source_ip": 1, "bad privilege": 1,
    }
    assert len(events) == 1 and events.iloc[0].source_ip == "198.51.100.7"


def test_ground_truth_columns_are_refused():
    leaky = io.StringIO(HEADER + ",label\n2026-10-04T10:00:00Z,10.0.1.5,,,,FAILED_LOGIN,FAILURE,22,,,,,benign\n")
    with pytest.raises(LabelLeakError):
        load_events(leaky)


def test_hosts_file_is_plain_inventory_only():
    assert load_hosts(io.StringIO("host,ip\napp01,10.0.2.1\n")) == {"app01": "10.0.2.1"}
    with pytest.raises(LabelLeakError):
        load_hosts(io.StringIO("host,ip,label\napp01,10.0.2.1,benign\n"))
    with pytest.raises(ValueError):
        load_hosts(io.StringIO("host,ip,owner\napp01,10.0.2.1,ops\n"))


def test_detection_package_never_reads_ground_truth_files():
    for path in (Path(__file__).parents[1] / "intruder").glob("*.py"):
        assert "ground_truth" not in path.read_text(encoding="utf-8"), path.name


# --- features ---------------------------------------------------------------------------------

def test_login_counts_distinct_users_and_failure_ratio():
    f = features(
        "2026-10-04T10:00:10Z,10.0.1.5,10.0.2.1,app01,ana,FAILED_LOGIN,FAILURE,22,,,,",
        "2026-10-04T10:00:20Z,10.0.1.5,10.0.2.1,app01,ana,FAILED_LOGIN,FAILURE,22,,,,",
        "2026-10-04T10:00:30Z,10.0.1.5,10.0.2.1,app01,bo,FAILED_LOGIN,FAILURE,22,,,,",
        "2026-10-04T10:00:40Z,10.0.1.5,10.0.2.1,app01,ana,SUCCESS_LOGIN,SUCCESS,22,,,,",
    )
    row = f.loc[("10.0.1.5", pd.Timestamp("2026-10-04 10:00"))]
    assert (row.failed_logins, row.successful_logins, row.distinct_users, row.event_count) == (3, 1, 2, 4)
    assert row.failure_ratio == pytest.approx(0.75)
    assert row.event_rate == pytest.approx(4 / 5)


def test_window_boundaries_are_half_open():
    f = features(
        "2026-10-04T10:04:59Z,10.0.1.5,10.0.2.1,app01,ana,FAILED_LOGIN,FAILURE,22,,,,",
        "2026-10-04T10:05:00Z,10.0.1.5,10.0.2.1,app01,ana,FAILED_LOGIN,FAILURE,22,,,,",
    )
    assert list(f.index.get_level_values("window_start")) == [pd.Timestamp("2026-10-04 10:00"),
                                                             pd.Timestamp("2026-10-04 10:05")]
    assert f.failed_logins.tolist() == [1, 1]


def test_ports_denials_and_web_counts():
    f = features(
        "2026-10-04T10:00:01Z,203.0.113.9,10.0.2.1,,,CONNECTION,DENY,21,0,,,",
        "2026-10-04T10:00:02Z,203.0.113.9,10.0.2.1,,,CONNECTION,DENY,23,0,,,",
        "2026-10-04T10:00:03Z,203.0.113.9,10.0.2.1,,,CONNECTION,ALLOW,443,500,,,",
        "2026-10-04T10:00:04Z,203.0.113.9,10.0.2.1,,,CONNECTION,ALLOW,443,500,,,",
        "2026-10-04T10:00:05Z,203.0.113.9,10.0.2.1,,,WEB_REQUEST,SUCCESS,443,200,404,SUSPECT_PROBE,",
        "2026-10-04T10:00:06Z,203.0.113.9,10.0.2.1,,,WEB_REQUEST,SUCCESS,443,900,200,NONE,",
    )
    row = f.iloc[0]
    assert (row.distinct_ports, row.denied_connections) == (3, 2)
    assert (row.web_requests, row.web_4xx, row.suspicious_requests) == (2, 1, 1)
    assert row.failure_ratio == pytest.approx(3 / 6)


def test_outbound_bytes_split_by_destination_class_and_denied_ignored():
    f = features(
        "2026-10-04T10:00:01Z,10.0.3.7,10.0.9.9,,,INTERNAL_TRANSFER,ALLOW,873,1000,,,",
        "2026-10-04T10:00:02Z,10.0.3.7,198.51.100.20,,,EXTERNAL_TRANSFER,ALLOW,443,250,,,",
        "2026-10-04T10:00:03Z,10.0.3.7,198.51.100.20,,,EXTERNAL_TRANSFER,DENY,443,9999,,,",
    )
    row = f.iloc[0]
    assert (row.outbound_internal_bytes, row.outbound_external_bytes) == (1000, 250)


def test_host_local_events_follow_the_login_session():
    cfg = Config(inventory={"app01": "10.0.2.1"})
    f = features(
        "2026-10-04T10:00:00Z,10.0.1.5,10.0.2.1,app01,ana,SUCCESS_LOGIN,SUCCESS,22,,,,",
        "2026-10-04T10:02:00Z,,,app01,ana,SUDO_EVENT,SUCCESS,,,,,ROUTINE",
        "2026-10-04T10:03:00Z,,,app01,ana,SUDO_EVENT,SUCCESS,,,,,ELEVATED_SHELL",
        "2026-10-04T10:04:00Z,,,app01,svc2,NEW_ACCOUNT,SUCCESS,,,,,",
        "2026-10-04T10:06:00Z,,,app01,carl,SUDO_EVENT,SUCCESS,,,,,ROUTINE",   # no session -> host ip
        cfg=cfg,
    )
    session = f.loc[("10.0.1.5", pd.Timestamp("2026-10-04 10:00"))]
    assert (session.sudo_routine, session.sudo_elevated, session.new_accounts) == (1, 1, 1)
    assert f.loc[("10.0.2.1", pd.Timestamp("2026-10-04 10:05"))].sudo_routine == 1


def test_session_attribution_expires():
    f = features(
        "2026-10-04T01:00:00Z,10.0.1.5,10.0.2.1,app01,ana,SUCCESS_LOGIN,SUCCESS,22,,,,",
        "2026-10-04T12:00:00Z,,,app01,ana,SUDO_EVENT,SUCCESS,,,,,ROUTINE",
    )
    assert "host:app01" in f.index.get_level_values("entity")


def test_off_hours_flag_and_model_matrix():
    f = features(
        "2026-10-04T03:00:00Z,10.0.1.5,10.0.2.1,app01,ana,SUCCESS_LOGIN,SUCCESS,22,,,,",
        "2026-10-04T11:00:00Z,10.0.1.5,10.0.2.1,app01,ana,SUCCESS_LOGIN,SUCCESS,22,,,,",
    )
    assert f.off_hours.tolist() == [1, 0]
    x = model_matrix(f.reset_index())
    assert x.shape == (2, len(FEATURES)) and np.isfinite(x).all()
