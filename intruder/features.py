"""5-minute behavioural features per entity (source IP).

Only observable event properties are used. Destination class (internal / external) is context, not a stage:
EXFIL is decided later from volume vs. baseline plus temporal context.
"""
from __future__ import annotations

from ipaddress import ip_address, ip_network

import numpy as np
import pandas as pd

from .config import Config
from .events import guard_labels

FEATURES = [
    "failed_logins", "successful_logins", "distinct_users", "distinct_ports", "denied_connections",
    "web_requests", "web_4xx", "suspicious_requests", "outbound_internal_bytes", "outbound_external_bytes",
    "sudo_routine", "sudo_elevated", "new_accounts", "event_count", "event_rate", "failure_ratio", "off_hours",
]
# Heavy-tailed counts/bytes are log-scaled before modelling; ratios and flags are not.
LOG_FEATURES = [f for f in FEATURES if f not in ("failure_ratio", "off_hours")]
NETWORK_EVENTS = ("CONNECTION", "EXTERNAL_TRANSFER", "INTERNAL_TRANSFER")


def _is_internal(value, nets) -> bool:
    try:
        return any(ip_address(value) in n for n in nets)
    except (TypeError, ValueError):
        return False


def attribute_actors(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Give every event an `entity` (the acting source IP).

    Host-local events without a source IP are attributed to the session that most plausibly caused them:
      SUDO_EVENT  -> latest SUCCESS_LOGIN of the same user on the same host within `session_hours`
      NEW_ACCOUNT -> latest attributed SUDO_EVENT on the same host within `account_link_min`
    Unattributable events fall back to the host's inventory IP, else "host:<name>".
    """
    df = events.copy()
    df["entity"] = df["source_ip"]
    df["attributed"] = False
    session = pd.Timedelta(hours=cfg.session_hours)
    link = pd.Timedelta(minutes=cfg.account_link_min)
    logins = df[(df.event_type == "SUCCESS_LOGIN") & df.host.notna() & df.username.notna()]

    for i in df.index[(df.event_type == "SUDO_EVENT") & df.source_ip.isna()]:
        row = df.loc[i]
        cand = logins[(logins.host == row.host) & (logins.username == row.username)
                      & (logins.ts <= row.ts) & (logins.ts >= row.ts - session)]
        if len(cand):
            df.at[i, "entity"], df.at[i, "attributed"] = cand.iloc[-1].source_ip, True

    for i in df.index[(df.event_type == "NEW_ACCOUNT") & df.source_ip.isna()]:
        row = df.loc[i]
        sudo = df[(df.event_type == "SUDO_EVENT") & (df.host == row.host) & df.entity.notna()
                  & (df.ts <= row.ts) & (df.ts >= row.ts - link)]
        if len(sudo):
            df.at[i, "entity"], df.at[i, "attributed"] = sudo.iloc[-1].entity, True

    missing = df.entity.isna()
    df.loc[missing, "entity"] = df.loc[missing, "host"].map(
        lambda h: cfg.inventory.get(h, f"host:{h}") if isinstance(h, str) else "unknown")
    return df


def build_features(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (entity, window_start) with the FEATURES columns."""
    guard_labels(events.columns)
    df = attribute_actors(events, cfg)
    nets = tuple(ip_network(n) for n in cfg.internal_nets)
    et = df.event_type
    net = et.isin(NETWORK_EVENTS)
    dst_internal = df.destination_ip.map(lambda v: _is_internal(v, nets))
    allowed = df.status != "DENY"
    login = et.isin(["FAILED_LOGIN", "SUCCESS_LOGIN"])

    df["window_start"] = df.ts.dt.floor(f"{cfg.window_min}min")
    df["failed_logins"] = (et == "FAILED_LOGIN").astype(int)
    df["successful_logins"] = (et == "SUCCESS_LOGIN").astype(int)
    df["login_user"] = df.username.where(login)
    df["conn_port"] = df.port.where(et == "CONNECTION")
    df["denied_connections"] = (net & ~allowed).astype(int)
    df["web_requests"] = (et == "WEB_REQUEST").astype(int)
    df["web_4xx"] = ((et == "WEB_REQUEST") & df.http_status.between(400, 499)).astype(int)
    df["suspicious_requests"] = ((et == "WEB_REQUEST") & (df.request_marker != "NONE")).astype(int)
    df["outbound_internal_bytes"] = df.bytes.where(net & allowed & dst_internal, 0)
    df["outbound_external_bytes"] = df.bytes.where(net & allowed & ~dst_internal & df.destination_ip.notna(), 0)
    df["sudo_routine"] = ((et == "SUDO_EVENT") & (df.privilege == "ROUTINE")).astype(int)
    df["sudo_elevated"] = ((et == "SUDO_EVENT") & (df.privilege == "ELEVATED_SHELL")).astype(int)
    df["new_accounts"] = (et == "NEW_ACCOUNT").astype(int)

    sums = ["failed_logins", "successful_logins", "denied_connections", "web_requests", "web_4xx",
            "suspicious_requests", "outbound_internal_bytes", "outbound_external_bytes", "sudo_routine",
            "sudo_elevated", "new_accounts"]
    g = df.groupby(["entity", "window_start"], sort=True)
    out = g[sums].sum()
    out["distinct_users"] = g["login_user"].nunique()
    out["distinct_ports"] = g["conn_port"].nunique()
    out["event_count"] = g.size()
    out["first_eid"] = g["eid"].min()
    out = out.reset_index()

    out["event_rate"] = out.event_count / cfg.window_min
    out["failure_ratio"] = (out.failed_logins + out.denied_connections + out.web_4xx) / out.event_count
    start, end = cfg.business_hours
    hour = out.window_start.dt.hour
    out["off_hours"] = ((hour < start) | (hour >= end)).astype(int)
    out["day"] = out.window_start.dt.date
    return out[["entity", "day", "window_start", "first_eid"] + FEATURES]


def model_matrix(features: pd.DataFrame) -> np.ndarray:
    """Feature matrix for the anomaly model: log1p on heavy-tailed columns."""
    x = features[FEATURES].astype(float).copy()
    x[LOG_FEATURES] = np.log1p(x[LOG_FEATURES].clip(lower=0))
    return x.to_numpy()
