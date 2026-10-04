"""Cached access to the pipeline outputs plus small, presentation-only derivations.

Everything here reads files written by train.py / predict.py / `py -m evaluation.run`. Ground truth is loaded only
for the Evaluation page and the explicitly labelled reveal box. Nothing here changes a detector decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from intruder.contracts import DETECTORS, STAGES
from intruder.features import FEATURES, attribute_actors, build_features
from intruder.prediction import ANOMALY_NORMAL_MAX
from train import load_config, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
KEYS = ["entity", "day"]

SPLITS = {"test_id": "Test (ID)", "test_ood": "Test (OOD)", "val": "Validation"}
SPLIT_HELP = {"test_id": "Test data with the same general behavioural profile as training.",
              "test_ood": "Test data with deliberately shifted benign behaviour.",
              "val": "Validation data used during development."}
SHORT = dict(zip(DETECTORS, "ABC"))
DETECTOR_NAME = {"A_anomaly_only": "Behaviour Anomaly", "B_stage_count": "Attack Signals",
                 "C_progression": "Attack Progression"}
DETECTOR_DESC = {"A_anomaly_only": "Is this behaviour unusual?",
                 "B_stage_count": "Is it unusual, and is there evidence of attack activity?",
                 "C_progression": "Do the attack signals form a plausible progression over time?"}
DEFAULT_DETECTOR = "C_progression"

STAGE_NAME = {"RECON": "Reconnaissance", "CRED_ATTACK": "Credential attack", "COMPROMISE": "Compromise",
              "PRIV_ESC": "Privilege escalation", "PERSISTENCE": "Persistence", "EXFIL": "Exfiltration"}
STAGE_TEXT = {"RECON": "The system observed reconnaissance-like activity.",
              "CRED_ATTACK": "Credential attack activity was observed.",
              "COMPROMISE": "Evidence consistent with successful access was observed.",
              "PRIV_ESC": "Privilege escalation activity was observed.",
              "PERSISTENCE": "Persistence-related activity was observed.",
              "EXFIL": "Outbound data transfer activity was observed."}
VERDICTS = ["Intrusion", "Failed attack", "Suspicious", "Normal"]
VERDICT_MEANING = {
    "Intrusion": "Attack signals reach the later stages of an attack (compromise or beyond), or several stages occur in "
                 "attack order, with enough supporting evidence.",
    "Failed attack": "Attack-like activity was observed, but the evidence did not show successful progression beyond "
                     "the initial stages.",
    "Suspicious": "Behaviour is unusual or shows an isolated attack signal, without a clear attack pattern.",
    "Normal": "Nothing unusual enough to raise an alert.",
}

FEATURE_LABEL = {
    "failed_logins": "Failed logins", "successful_logins": "Successful logins", "distinct_users": "Distinct users",
    "failure_ratio": "Failure ratio", "distinct_ports": "Distinct ports", "denied_connections": "Denied connections",
    "outbound_internal_bytes": "Internal outbound traffic", "outbound_external_bytes": "External outbound traffic",
    "web_requests": "Web requests", "web_4xx": "Web 4xx responses", "suspicious_requests": "Suspicious requests",
    "sudo_routine": "Routine sudo", "sudo_elevated": "Elevated sudo", "new_accounts": "New accounts",
    "event_count": "Event count", "event_rate": "Event rate (per minute)", "off_hours": "Off-hours activity",
}
FEATURE_GROUPS = {
    "Access & authentication": ["failed_logins", "successful_logins", "distinct_users", "failure_ratio"],
    "Network activity": ["distinct_ports", "denied_connections", "outbound_internal_bytes", "outbound_external_bytes"],
    "Web activity": ["web_requests", "web_4xx", "suspicious_requests"],
    "Privilege & behaviour": ["sudo_routine", "sudo_elevated", "new_accounts", "event_count", "event_rate", "off_hours"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES)


# ------------------------------------------------------------------------------------------------ loaders
def ready() -> bool:
    return (RESULTS / "predictions.csv").exists() and (RESULTS / "metrics.json").exists()


@st.cache_data(show_spinner="Loading detection results")
def predictions() -> pd.DataFrame:
    p = pd.read_csv(RESULTS / "predictions.csv", dtype=str, keep_default_na=False)
    p["risk_score"] = pd.to_numeric(p.risk_score)
    p["alert"] = p.alert.eq("True")
    p["stages_completed"] = pd.to_numeric(p.stages_completed).astype(int)
    p["first_warning_ts"] = pd.to_datetime(p.first_warning_ts.replace("", None))
    return p


@st.cache_data(show_spinner="Loading detector outputs")
def outputs(split: str):
    d = RESULTS / split
    ws = pd.read_csv(d / "window_scores.csv", parse_dates=["window_start"])
    ws["day"] = ws.window_start.dt.date.astype(str)
    sig = pd.read_csv(d / "stage_signals.csv", parse_dates=["ts"])
    sig["day"] = sig.ts.dt.date.astype(str)
    prog = pd.read_csv(d / "progression.csv", dtype=str, keep_default_na=False)
    prog["progression_score"] = pd.to_numeric(prog.progression_score)
    prog["day"] = pd.to_datetime(prog.day).dt.date.astype(str)
    return ws, sig, prog


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def training_summary() -> dict:
    path = MODELS / "training_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner="Linking events to entities")
def attributed_events(split: str) -> pd.DataFrame:
    cfg = load_config(DATA)
    events, _ = load_split(DATA, split, cfg)
    ev = attribute_actors(events, cfg)
    ev["day"] = ev.ts.dt.date.astype(str)
    return ev


@st.cache_data(show_spinner="Building behaviour profiles")
def features(split: str) -> pd.DataFrame:
    cfg = load_config(DATA)
    events, _ = load_split(DATA, split, cfg)
    f = build_features(events, cfg)
    f["day"] = f.day.astype(str)
    return f


@st.cache_data(show_spinner="Learning the benign baseline")
def baseline() -> pd.DataFrame:
    """Median and robust scale of each feature on the benign training windows (MAD, falling back to std)."""
    f = features("train")[FEATURES].astype(float)
    median = f.median()
    mad = (f - median).abs().median() * 1.4826
    scale = mad.where(mad > 0, f.std()).where(lambda s: s > 0, 1.0)
    return pd.DataFrame({"median": median, "scale": scale})


@st.cache_data(show_spinner=False)
def ground_truth(split: str) -> pd.DataFrame:
    from evaluation.ground_truth import load_ground_truth  # evaluation-only data
    gt = load_ground_truth(DATA / split / "ground_truth.csv")
    gt["day"] = gt.day.astype(str)
    return gt


# ------------------------------------------------------------------------------------------------ derived views
@st.cache_data(show_spinner="Preparing the investigation view")
def entity_days(split: str) -> pd.DataFrame:
    """One row per entity-day with every detector's verdict side by side plus stage and timing context."""
    p = predictions()
    p = p[p.split == split]
    cols = ["risk_score", "predicted_class", "alert", "first_warning_ts", "explanation", "stages_completed",
            "current_stage"]
    table = None
    for det in DETECTORS:
        s = SHORT[det]
        d = p[p.detector == det][KEYS + cols].rename(columns={c: f"{s}_{c}" for c in cols})
        table = d if table is None else table.merge(d, on=KEYS, how="outer")

    ws, sig, prog = outputs(split)
    win = ws.groupby(KEYS).agg(anomaly=("anomaly_score", "max"), first_activity=("window_start", "min"))
    table = table.join(win, on=KEYS)
    if len(sig):
        raw = sig.groupby(KEYS).agg(b_stages=("stage", lambda s: sorted(set(s), key=STAGES.index)),
                                    signals=("stage", "size"))
        table = table.join(raw, on=KEYS)
    else:
        table["b_stages"], table["signals"] = None, 0
    prog = prog.assign(c_stages=prog.stages_detected.map(lambda v: list(dict.fromkeys(x for x in v.split("|") if x))))
    table = table.merge(prog[KEYS + ["c_stages", "progression_score", "next_expected_stage"]], on=KEYS, how="left")

    for col in ("b_stages", "c_stages"):
        table[col] = table[col].map(lambda v: v if isinstance(v, list) else [])
    table = table.fillna({"signals": 0, "progression_score": 0.0, "next_expected_stage": "", "anomaly": 0.0})
    table["signals"] = table.signals.astype(int)
    for det in DETECTORS:
        s = SHORT[det]
        table[f"{s}_latency_s"] = (table[f"{s}_first_warning_ts"] - table.first_activity).dt.total_seconds()
    table["any_alert"] = table[[f"{SHORT[d]}_alert" for d in DETECTORS]].any(axis=1)
    return table.sort_values(["C_risk_score", "B_risk_score", "anomaly"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def question_counts(split: str) -> dict:
    """Entity-days answering each of the detector's questions (each count is independent, not a strict funnel)."""
    t = entity_days(split)
    return {"total": len(t), "unusual": int((t.anomaly >= ANOMALY_NORMAL_MAX).sum()),
            "signals": int((t.signals > 0).sum()), "ordered": int((t.c_stages.map(len) >= 2).sum()),
            "intrusion": int((t.C_predicted_class == "Intrusion").sum())}


@st.cache_data(show_spinner=False)
def stage_activity(split: str) -> pd.DataFrame:
    """Observed attack-stage signals: number of signals and of entity-days per stage (all six stages listed)."""
    _, sig, _ = outputs(split)
    rows = []
    for stage in STAGES:
        s = sig[sig.stage == stage]
        rows.append({"stage": stage, "signals": len(s), "entities": s.drop_duplicates(KEYS).shape[0]})
    return pd.DataFrame(rows)


def episodes(sig: pd.DataFrame) -> list[dict]:
    """Consecutive signals of the same stage grouped into one step of the story."""
    out: list[dict] = []
    for r in sig.sort_values(["ts", "eid"]).itertuples():
        if out and out[-1]["stage"] == r.stage:
            out[-1].update(end=r.ts, count=out[-1]["count"] + 1, evidence=r.evidence)
        else:
            out.append({"stage": r.stage, "start": r.ts, "end": r.ts, "count": 1, "evidence": r.evidence})
    return out


def window_profile(split: str, entity: str, window_start) -> pd.Series | None:
    """Feature values of one five-minute processing window, or None."""
    f = features(split)
    row = f[(f.entity == entity) & (f.window_start == pd.Timestamp(window_start))]
    return None if row.empty else row.iloc[0]


def deviations(split: str, entity: str, day: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Feature values of the entity-day's most unusual window next to the benign training baseline."""
    ws, _, _ = outputs(split)
    w = ws[(ws.entity == entity) & (ws.day == day)].sort_values("anomaly_score", ascending=False)
    if w.empty:
        return pd.DataFrame(), None
    top = w.iloc[0]
    row = window_profile(split, entity, top.window_start)
    if row is None:
        return pd.DataFrame(), top
    base = baseline()
    out = pd.DataFrame({"feature": FEATURES, "value": [float(row[c]) for c in FEATURES]})
    out["typical"] = base["median"].values
    out["deviation"] = (out.value - out.typical).abs() / base["scale"].values
    return out, top


# ------------------------------------------------------------------------------------------------ formatting
def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


def feature_value(feature: str, value: float) -> str:
    if feature.endswith("_bytes"):
        return fmt_bytes(value)
    if feature == "failure_ratio":
        return f"{value:.0%}"
    if feature == "event_rate":
        return f"{value:,.1f}"
    if feature == "off_hours":
        return "Yes" if value else "No"
    return f"{value:,.0f}"


def fmt_duration(seconds) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    s = int(round(seconds))
    sign = "−" if s < 0 else ""
    s = abs(s)
    if s < 60:
        return f"{sign}{s} s"
    if s < 3600:
        return f"{sign}{s // 60} min {s % 60:02d} s" if s % 60 else f"{sign}{s // 60} min"
    return f"{sign}{s // 3600} h {s % 3600 // 60:02d} min"


def fmt_time(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%H:%M:%S")


def fmt_hm(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%H:%M")


def fmt_day(day: str) -> str:
    return pd.Timestamp(day).strftime("%d %b %Y")


def fmt_pct(v) -> str:
    return "—" if v is None else f"{v:.1%}"


def fmt_num(v, digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


# ------------------------------------------------------------------------------------------------ session state
def current_split() -> str:
    return st.session_state.get("split", "test_id")


def focus_row(table: pd.DataFrame) -> pd.Series:
    """The entity-day under investigation; defaults to the highest-risk entity-day of the dataset."""
    split = current_split()
    focus = st.session_state.get("focus")
    if focus and focus[0] == split:
        hit = table[(table.entity == focus[1]) & (table.day == focus[2])]
        if len(hit):
            return hit.iloc[0]
    row = table.iloc[0]
    st.session_state["focus"] = (split, row.entity, row.day)
    return row


def set_focus(entity: str, day: str) -> None:
    st.session_state["focus"] = (current_split(), entity, day)
