"""Turn raw log files (syslog/auth.log, web access logs, CSV / JSON-lines) into one normalized events table.

Every event keeps its original line, file name and line number so findings can always point back to evidence.
Lines that cannot be understood are counted and sampled instead of crashing ingestion.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import PurePath
from urllib.parse import unquote_plus

import pandas as pd

COLUMNS = [
    "ts", "source", "host", "src_ip", "dst_ip", "dst_port", "user", "event_type", "status",
    "bytes", "method", "path", "http_status", "user_agent", "detail", "file", "line_no", "raw",
]

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

# --- syslog / auth.log -------------------------------------------------------------------------
SYSLOG_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+"
    r"(?P<prog>[\w\-./]+?)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*)$")
ISO_SYSLOG_RE = re.compile(
    r"^(?P<iso>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+(?P<host>\S+)\s+"
    r"(?P<prog>[\w\-./]+?)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*)$")
SSH_FAILED = re.compile(
    r"Failed (?P<method>\S+) for (?P<invalid>invalid user )?(?P<user>\S+) from (?P<ip>[0-9a-fA-F.:]+)(?: port (?P<port>\d+))?")
SSH_ACCEPTED = re.compile(r"Accepted (?P<method>\S+) for (?P<user>\S+) from (?P<ip>[0-9a-fA-F.:]+)(?: port (?P<port>\d+))?")
SSH_INVALID = re.compile(r"Invalid user (?P<user>\S*) from (?P<ip>[0-9a-fA-F.:]+)")
SUDO_RE = re.compile(r"^\s*(?P<user>\S+)\s*:.*?USER=(?P<target>\S+)\s*;\s*COMMAND=(?P<cmd>.*)$")
USERADD_RE = re.compile(r"new user: name=(?P<user>[^,\s]+)")

# --- web access log (common / combined) --------------------------------------------------------
ACCESS_RE = re.compile(
    r'^(?P<ip>\S+) \S+ (?P<user>\S+) \[(?P<ts>[^\]]+)\] "(?P<method>[A-Z]+) (?P<path>\S+)(?: [^"]*)?" '
    r'(?P<status>\d{3}) (?P<size>\d+|-)(?: "(?P<ref>[^"]*)" "(?P<ua>[^"]*)")?')
ACCESS_TS_RE = re.compile(r"(\d{1,2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})(?:\s*([+-])(\d{2})(\d{2}))?")

# --- tabular (CSV / JSON lines) column aliases -------------------------------------------------
ALIASES = {
    "ts": ["timestamp", "time", "ts", "datetime", "date", "@timestamp", "eventtime", "event_time"],
    "src_ip": ["src_ip", "source_ip", "src", "srcaddr", "client_ip", "ip", "source", "remote_addr"],
    "dst_ip": ["dst_ip", "dest_ip", "destination_ip", "dst", "dstaddr", "destination"],
    "dst_port": ["dst_port", "dest_port", "dport", "destination_port", "port"],
    "bytes": ["bytes", "bytes_out", "sent_bytes", "bytes_sent", "size", "length"],
    "status": ["action", "verdict", "disposition", "status", "outcome", "result"],
    "user": ["user", "username", "user_name", "account"],
    "host": ["host", "hostname", "device", "computer"],
    "detail": ["message", "msg", "event", "event_type", "description"],
}


@dataclass
class FileReport:
    name: str
    format: str = "unknown"
    parsed: int = 0
    skipped: int = 0
    examples: list[str] = field(default_factory=list)  # a few lines we could not parse

    def skip(self, line_no: int, line: str) -> None:
        self.skipped += 1
        if len(self.examples) < 5:
            self.examples.append(f"line {line_no}: {line[:160]}")


def _naive_utc(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def parse_syslog_line(line: str, year: int) -> dict | None:
    m = SYSLOG_RE.match(line) or ISO_SYSLOG_RE.match(line)
    if not m:
        return None
    g = m.groupdict()
    try:
        if g.get("iso"):
            ts = _naive_utc(pd.Timestamp(g["iso"]).to_pydatetime())
        else:
            h, mi, s = map(int, g["time"].split(":"))
            ts = datetime(year, MONTHS[g["mon"]], int(g["day"]), h, mi, s)
    except (KeyError, ValueError):
        return None

    prog, msg = g["prog"], g["msg"]
    ev = {"ts": ts, "source": "auth", "host": g["host"], "event_type": "other", "detail": msg}
    if prog.startswith("sshd"):
        if f := SSH_FAILED.search(msg):
            ev.update(event_type="auth_failure", status="failure", user=f["user"], src_ip=f["ip"],
                      dst_port=22, method=f["method"],
                      detail=f"SSH {f['method']} failed for {'invalid user ' if f['invalid'] else ''}{f['user']}")
        elif a := SSH_ACCEPTED.search(msg):
            ev.update(event_type="auth_success", status="success", user=a["user"], src_ip=a["ip"],
                      dst_port=22, method=a["method"], detail=f"SSH login accepted ({a['method']}) for {a['user']}")
        elif i := SSH_INVALID.search(msg):
            ev.update(event_type="invalid_user", user=i["user"], src_ip=i["ip"], detail=f"Probe for non-existent user {i['user']}")
    elif prog == "sudo":
        if s := SUDO_RE.match(msg):
            ev.update(event_type="sudo", status="success", user=s["user"], detail=f"sudo as {s['target']}: {s['cmd'].strip()}",
                      path=s["cmd"].strip(), method=s["target"])
    elif prog in ("useradd", "adduser"):
        if u := USERADD_RE.search(msg):
            ev.update(event_type="user_created", user=u["user"], detail=f"New local account created: {u['user']}")
    return ev


def _access_ts(text: str) -> datetime | None:
    m = ACCESS_TS_RE.search(text)
    if not m or m[2] not in MONTHS:
        return None
    ts = datetime(int(m[3]), MONTHS[m[2]], int(m[1]), int(m[4]), int(m[5]), int(m[6]))
    if m[7]:
        offset = timedelta(hours=int(m[8]), minutes=int(m[9]))
        ts = ts - offset if m[7] == "+" else ts + offset
    return ts


def parse_access_line(line: str) -> dict | None:
    m = ACCESS_RE.match(line)
    if not m:
        return None
    ts = _access_ts(m["ts"])
    if ts is None:
        return None
    path = unquote_plus(m["path"])
    return {
        "ts": ts, "source": "web", "src_ip": m["ip"], "user": None if m["user"] == "-" else m["user"],
        "event_type": "http_request", "method": m["method"], "path": path, "http_status": int(m["status"]),
        "bytes": 0 if m["size"] == "-" else int(m["size"]), "user_agent": m["ua"] or "",
        "status": "success" if int(m["status"]) < 400 else "failure",
        "detail": f"{m['method']} {path} -> {m['status']}",
    }


def _pick(columns: list[str], key: str) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for alias in ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def parse_table(frame: pd.DataFrame, report: FileReport) -> list[dict]:
    """Map an arbitrary tabular log onto the event schema using column-name aliases."""
    cols = {k: _pick(list(frame.columns), k) for k in ALIASES}
    if cols["ts"] is None:
        report.skip(1, "no timestamp column found; columns = " + ", ".join(map(str, frame.columns)))
        report.skipped = len(frame)
        return []
    times = pd.to_datetime(frame[cols["ts"]], errors="coerce", utc=True).dt.tz_localize(None)
    is_net = cols["dst_port"] is not None or cols["dst_ip"] is not None
    events = []
    for pos, (idx, row) in enumerate(frame.iterrows()):
        line_no = pos + 2  # header is line 1
        ts = times.iloc[pos]
        if pd.isna(ts):
            report.skip(line_no, ",".join(map(str, row.values)))
            continue
        get = lambda k: (None if cols[k] is None or pd.isna(row[cols[k]]) else row[cols[k]])  # noqa: E731
        status = str(get("status") or "").lower()
        ev = {"ts": ts.to_pydatetime(), "src_ip": get("src_ip"), "user": get("user"), "host": get("host"),
              "detail": get("detail"), "raw": ",".join(map(str, row.values)), "line_no": line_no}
        if is_net:
            port, size = get("dst_port"), get("bytes")
            ev.update(source="net", event_type="net_conn", dst_ip=get("dst_ip"),
                      dst_port=pd.to_numeric(port, errors="coerce"), bytes=pd.to_numeric(size, errors="coerce"),
                      status="deny" if any(w in status for w in ("deny", "drop", "block", "reject")) else "allow")
            ev["detail"] = ev["detail"] or f"{ev['src_ip']} -> {ev['dst_ip']}:{port} {ev['status']} ({size} bytes)"
        else:
            ok = any(w in status for w in ("success", "accept", "ok", "pass", "allow"))
            ev.update(source="auth", event_type="auth_success" if ok else "auth_failure",
                      status="success" if ok else "failure")
        events.append(ev)
    return events


def parse_file(name: str, data: bytes, year: int) -> tuple[list[dict], FileReport]:
    report = FileReport(name=name)
    text = data.decode("utf-8", errors="replace").lstrip("﻿")
    suffix = PurePath(name).suffix.lower()
    first = next((ln for ln in text.splitlines() if ln.strip()), "")

    if suffix in (".json", ".jsonl", ".ndjson") or first.lstrip().startswith("{"):
        report.format = "json-lines"
        rows, positions = [], []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
                positions.append(n)
            except json.JSONDecodeError:
                report.skip(n, line)
        if not rows:
            return [], report
        events = parse_table(pd.DataFrame(rows), report)
        for ev in events:  # restore true line numbers
            ev["line_no"] = positions[ev["line_no"] - 2]
    elif suffix in (".csv", ".tsv"):
        report.format = "csv"
        sep = "\t" if suffix == ".tsv" else ","
        try:
            frame = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip", dtype=str)
        except Exception as exc:  # pragma: no cover - pandas raises many types here
            report.skip(1, f"unreadable table: {exc}")
            return [], report
        events = parse_table(frame, report)
    else:
        events, kinds = [], set()
        for n, line in enumerate(text.splitlines(), 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            ev = parse_access_line(line)
            if ev:
                kinds.add("web access log")
            else:
                ev = parse_syslog_line(line, year)
                if ev:
                    kinds.add("syslog/auth")
            if ev is None:
                report.skip(n, line)
                continue
            ev.update(raw=line, line_no=n)
            events.append(ev)
        report.format = " + ".join(sorted(kinds)) or "unrecognised"

    for ev in events:
        ev["file"] = name
    report.parsed = len(events)
    return events, report


def load(files: list[tuple[str, bytes]], year: int) -> tuple[pd.DataFrame, list[FileReport]]:
    all_events, reports = [], []
    for name, data in files:
        events, report = parse_file(name, data, year)
        all_events.extend(events)
        reports.append(report)
    df = pd.DataFrame(all_events, columns=COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"])
    for col in ("dst_port", "bytes", "http_status", "line_no"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("src_ip", "dst_ip", "user", "host"):
        df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip())
    df = df.sort_values(["ts", "file", "line_no"], kind="stable").reset_index(drop=True)
    df.insert(0, "eid", range(len(df)))
    return df, reports
