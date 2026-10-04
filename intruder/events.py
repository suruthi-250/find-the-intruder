"""Load and validate abstract security events (events.csv) and the host inventory (hosts.csv).

Rows that violate the schema are skipped and counted by reason instead of crashing the pipeline.
Ground-truth columns are refused outright so evaluation labels can never leak into detection.
"""
from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network
from pathlib import Path

import pandas as pd

EVENT_TYPES = {"FAILED_LOGIN", "SUCCESS_LOGIN", "WEB_REQUEST", "CONNECTION", "SUDO_EVENT", "NEW_ACCOUNT",
               "EXTERNAL_TRANSFER", "INTERNAL_TRANSFER"}
STATUSES = {"SUCCESS", "FAILURE", "ALLOW", "DENY"}
MARKERS = {"NONE", "SUSPECT_SQLI", "SUSPECT_PATH_TRAVERSAL", "SUSPECT_PROBE"}
PRIVILEGES = {"ROUTINE", "ELEVATED_SHELL"}
# Host-local events may omit source_ip; features attribute them to the login session instead.
HOST_LOCAL = {"SUDO_EVENT", "NEW_ACCOUNT"}

# Only synthetic address space is accepted: RFC 1918 internal and RFC 5737 documentation ranges.
SYNTHETIC_EXTERNAL = tuple(ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))
# Columns that belong to ground truth. Their presence means labels are being mixed into detection input.
LABEL_COLUMNS = {"label", "trap_type", "expected_stages", "attack_start", "attack_end", "scenario_id", "stage"}

COLUMNS = ["timestamp", "source_ip", "destination_ip", "host", "username", "event_type", "status", "port",
           "bytes", "http_status", "request_marker", "privilege"]
REQUIRED = ["timestamp", "event_type", "status"]


class LabelLeakError(ValueError):
    """Raised when detection input carries ground-truth columns."""


@dataclass
class LoadReport:
    source: str
    rows: int = 0
    valid: int = 0
    skipped: int = 0
    reasons: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)


def _ip_ok(value: str, internal_nets) -> bool:
    try:
        ip = ip_address(value)
    except ValueError:
        return False
    return any(ip in n for n in internal_nets) or any(ip in n for n in SYNTHETIC_EXTERNAL)


def _read(source) -> tuple[pd.DataFrame, str]:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, dtype=str, keep_default_na=False), str(source)
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    return pd.read_csv(source, dtype=str, keep_default_na=False), getattr(source, "name", "<upload>")


def guard_labels(columns) -> None:
    leaked = LABEL_COLUMNS & {str(c).strip().lower() for c in columns}
    if leaked:
        raise LabelLeakError(f"detection input contains ground-truth columns: {sorted(leaked)}")


def load_events(source, internal_nets=("10.0.0.0/8",)) -> tuple[pd.DataFrame, LoadReport]:
    raw, name = _read(source)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    guard_labels(raw.columns)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    for col in COLUMNS:
        if col not in raw.columns:
            raw[col] = ""
    df = raw[COLUMNS].apply(lambda s: s.str.strip())
    df["line_no"] = range(2, len(df) + 2)  # header is line 1
    report = LoadReport(source=name, rows=len(df))
    nets = tuple(ip_network(n) for n in internal_nets)

    df["event_type"] = df["event_type"].str.upper()
    df["status"] = df["status"].str.upper()
    df["request_marker"] = df["request_marker"].str.upper().replace("", "NONE")
    df["privilege"] = df["privilege"].str.upper()
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="ISO8601")
    port = pd.to_numeric(df["port"].replace("", None), errors="coerce")
    size = pd.to_numeric(df["bytes"].replace("", None), errors="coerce")
    http = pd.to_numeric(df["http_status"].replace("", None), errors="coerce")

    problems = {
        "bad timestamp": ts.isna(),
        "unknown event_type": ~df["event_type"].isin(EVENT_TYPES),
        "unknown status": ~df["status"].isin(STATUSES),
        "missing source_ip": (df["source_ip"] == "") & ~df["event_type"].isin(HOST_LOCAL),
        "invalid or non-synthetic source_ip": (df["source_ip"] != "") & ~df["source_ip"].map(lambda v: _ip_ok(v, nets)),
        "invalid or non-synthetic destination_ip":
            (df["destination_ip"] != "") & ~df["destination_ip"].map(lambda v: _ip_ok(v, nets)),
        "bad port": (df["port"] != "") & (port.isna() | (port < 0) | (port > 65535)),
        "bad bytes": (df["bytes"] != "") & (size.isna() | (size < 0)),
        "bad http_status": (df["http_status"] != "") & (http.isna() | (http < 100) | (http > 599)),
        "unknown request_marker": ~df["request_marker"].isin(MARKERS),
        "bad privilege": ((df["event_type"] == "SUDO_EVENT") & ~df["privilege"].isin(PRIVILEGES))
                         | ((df["event_type"] != "SUDO_EVENT") & (df["privilege"] != "")),
    }
    bad = pd.Series(False, index=df.index)
    for reason, mask in problems.items():
        new = mask & ~bad  # count each row once, under its first problem
        report.reasons[reason] += int(new.sum())
        for i in df.index[new][: max(0, 5 - len(report.examples))]:
            report.examples.append(f"line {df.at[i, 'line_no']}: {reason}")
        bad |= mask
    report.reasons = Counter({k: v for k, v in report.reasons.items() if v})
    report.skipped, report.valid = int(bad.sum()), int((~bad).sum())

    out = df.loc[~bad].copy()
    out["ts"] = ts[~bad].dt.tz_localize(None)
    out["port"], out["bytes"], out["http_status"] = port[~bad], size[~bad].fillna(0), http[~bad]
    for col in ("source_ip", "destination_ip", "host", "username", "privilege"):
        out[col] = out[col].replace("", None)
    out = out.drop(columns="timestamp").sort_values(["ts", "line_no"], kind="stable").reset_index(drop=True)
    out.insert(0, "eid", range(len(out)))
    return out, report


def load_hosts(source) -> dict[str, str]:
    """hosts.csv is a plain host -> ip inventory; anything beyond those two columns is rejected."""
    raw, name = _read(source)
    cols = {str(c).strip().lower() for c in raw.columns}
    guard_labels(cols)
    if cols != {"host", "ip"}:
        raise ValueError(f"{name}: hosts file must have exactly the columns host,ip (got {sorted(cols)})")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    return {r.host.strip(): r.ip.strip() for r in raw.itertuples() if r.host.strip() and r.ip.strip()}
