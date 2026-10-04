"""Deterministic defensive behavioral-stage inference from synthetic event logs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .contracts import STAGES, STAGE_INDEX
from .features import attribute_actors


# These are deliberately conservative, deterministic thresholds.  They are
# behavioral indicators, not claims that a single event proves compromise.
RECON_WINDOW_MIN = 10
RECON_MIN_CONNECTIONS = 4
RECON_MIN_DISTINCT_PORTS = 4
RECON_MIN_DENIED = 3

CRED_WINDOW_MIN = 10
CRED_MIN_FAILURES = 3

COMPROMISE_WINDOW_MIN = 15
COMPROMISE_MIN_FAILURES = 2

EXFIL_WINDOW_HOURS = 6
EXFIL_MIN_BYTES = 500_000.0


STAGE_SIGNAL_COLUMNS = ["entity", "ts", "stage", "eid", "evidence"]
PROGRESSION_COLUMNS = [
    "entity",
    "day",
    "stages_detected",
    "stage_times",
    "stages_completed",
    "current_stage",
    "next_expected_stage",
    "progression_score",
    "first_warning_ts",
]


@dataclass(frozen=True)
class _Signal:
    entity: str
    ts: pd.Timestamp
    stage: str
    eid: int
    evidence: str


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=STAGE_SIGNAL_COLUMNS)


def _empty_progression() -> pd.DataFrame:
    return pd.DataFrame(columns=PROGRESSION_COLUMNS)


def _normalise_events(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return actor-attributed, chronologically ordered event records."""
    if events.empty:
        out = events.copy()
        if "entity" not in out.columns:
            out["entity"] = pd.Series(dtype="object")
        return out

    required = {"eid", "ts", "event_type", "status"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")

    out = events.copy()
    if "entity" not in out.columns or "attributed" not in out.columns:
        out = attribute_actors(out, cfg)

    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.loc[out["ts"].notna()].copy()
    out["event_type"] = out["event_type"].astype(str).str.upper()
    out["status"] = out["status"].astype(str).str.upper()
    out["eid"] = pd.to_numeric(out["eid"], errors="coerce").astype("Int64")
    out = out.loc[out["eid"].notna()].copy()
    out["eid"] = out["eid"].astype(int)
    out["entity"] = out["entity"].astype(str)
    return out.sort_values(["entity", "ts", "eid"], kind="stable").reset_index(drop=True)


def _window_mask(times: pd.Series, start: pd.Timestamp, minutes: float) -> pd.Series:
    return (times >= start) & (times <= start + pd.Timedelta(minutes=minutes))


def _add_signal(rows: list[_Signal], row: pd.Series, stage: str, evidence: str) -> None:
    if stage not in STAGE_INDEX:
        return
    rows.append(
        _Signal(
            entity=str(row["entity"]),
            ts=pd.Timestamp(row["ts"]),
            stage=stage,
            eid=int(row["eid"]),
            evidence=evidence.strip(),
        )
    )


def _detect_recon(group: pd.DataFrame, rows: list[_Signal]) -> None:
    connections = group[group["event_type"].eq("CONNECTION")].copy()
    if connections.empty:
        return

    ports = pd.to_numeric(connections.get("port"), errors="coerce")
    denied = connections["status"].eq("DENY")

    for idx, row in connections.iterrows():
        start = row["ts"] - pd.Timedelta(minutes=RECON_WINDOW_MIN)
        mask = (connections["ts"] >= start) & (connections["ts"] <= row["ts"])
        window = connections.loc[mask]
        port_values = pd.to_numeric(window["port"], errors="coerce").dropna()
        distinct_ports = int(port_values.nunique())
        denied_count = int(window["status"].eq("DENY").sum())
        connection_count = len(window)

        if (
            connection_count >= RECON_MIN_CONNECTIONS
            and (
                distinct_ports >= RECON_MIN_DISTINCT_PORTS
                or denied_count >= RECON_MIN_DENIED
            )
        ):
            _add_signal(
                rows,
                row,
                "RECON",
                f"{connection_count} connections, {distinct_ports} distinct ports, "
                f"{denied_count} denied in {RECON_WINDOW_MIN}m",
            )


def _detect_cred_attack(group: pd.DataFrame, rows: list[_Signal]) -> None:
    failures = group[
        group["event_type"].eq("FAILED_LOGIN")
        & group["status"].eq("FAILURE")
    ].copy()
    if failures.empty:
        return

    for _, row in failures.iterrows():
        start = row["ts"] - pd.Timedelta(minutes=CRED_WINDOW_MIN)
        mask = (failures["ts"] >= start) & (failures["ts"] <= row["ts"])
        count = int(mask.sum())
        if count >= CRED_MIN_FAILURES:
            _add_signal(
                rows,
                row,
                "CRED_ATTACK",
                f"{count} failed logins in {CRED_WINDOW_MIN}m",
            )


def _detect_compromise(group: pd.DataFrame, rows: list[_Signal]) -> None:
    failures = group[
        group["event_type"].eq("FAILED_LOGIN")
        & group["status"].eq("FAILURE")
    ].copy()
    successes = group[
        group["event_type"].eq("SUCCESS_LOGIN")
        & group["status"].eq("SUCCESS")
    ].copy()
    if failures.empty or successes.empty:
        return

    for _, success in successes.iterrows():
        start = success["ts"] - pd.Timedelta(minutes=COMPROMISE_WINDOW_MIN)
        recent = failures[
            (failures["ts"] >= start) & (failures["ts"] <= success["ts"])
        ]

        # Prefer the same username when it is available on both sides.
        username = success.get("username")
        if pd.notna(username) and str(username).strip():
            same_user = recent[
                recent["username"].fillna("").astype(str).eq(str(username))
            ]
            if len(same_user) >= COMPROMISE_MIN_FAILURES:
                recent = same_user

        if len(recent) >= COMPROMISE_MIN_FAILURES:
            _add_signal(
                rows,
                success,
                "COMPROMISE",
                f"successful login after {len(recent)} failed logins "
                f"in {COMPROMISE_WINDOW_MIN}m",
            )


def _detect_priv_esc(group: pd.DataFrame, rows: list[_Signal]) -> None:
    matches = group[
        group["event_type"].eq("SUDO_EVENT")
        & group["privilege"].fillna("").astype(str).str.upper().eq("ELEVATED_SHELL")
    ]
    for _, row in matches.iterrows():
        _add_signal(rows, row, "PRIV_ESC", "elevated shell sudo event")


def _detect_persistence(group: pd.DataFrame, rows: list[_Signal]) -> None:
    matches = group[group["event_type"].eq("NEW_ACCOUNT")]
    for _, row in matches.iterrows():
        account = str(row.get("username") or "").strip()
        detail = f"new account '{account}'" if account else "new account event"
        _add_signal(rows, row, "PERSISTENCE", detail)


def _detect_exfil(group: pd.DataFrame, rows: list[_Signal]) -> None:
    transfers = group[group["event_type"].eq("EXTERNAL_TRANSFER")].copy()
    if transfers.empty:
        return

    compromise_times = []
    # Use already-observable authentication evidence rather than a precomputed
    # label: a successful login preceded by repeated failures.
    failures = group[
        group["event_type"].eq("FAILED_LOGIN")
        & group["status"].eq("FAILURE")
    ].copy()
    successes = group[
        group["event_type"].eq("SUCCESS_LOGIN")
        & group["status"].eq("SUCCESS")
    ].copy()

    for _, success in successes.iterrows():
        start = success["ts"] - pd.Timedelta(minutes=COMPROMISE_WINDOW_MIN)
        recent = failures[
            (failures["ts"] >= start) & (failures["ts"] <= success["ts"])
        ]
        if len(recent) >= COMPROMISE_MIN_FAILURES:
            compromise_times.append(success["ts"])

    for _, row in transfers.iterrows():
        amount = float(pd.to_numeric(pd.Series([row.get("bytes", 0)]), errors="coerce").fillna(0).iloc[0])
        if amount < EXFIL_MIN_BYTES:
            continue

        recent_compromise = [
            ts
            for ts in compromise_times
            if ts <= row["ts"]
            and row["ts"] - ts <= pd.Timedelta(hours=EXFIL_WINDOW_HOURS)
        ]
        if recent_compromise:
            _add_signal(
                rows,
                row,
                "EXFIL",
                f"{amount:.0f} outbound bytes after suspicious authentication "
                f"within {EXFIL_WINDOW_HOURS}h",
            )


def detect_stage_signals(events: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Infer deterministic behavioral-stage signals from existing event logs."""
    cfg = cfg or Config()
    frame = _normalise_events(events, cfg)
    if frame.empty:
        return _empty_signals()

    rows: list[_Signal] = []
    for _, group in frame.groupby("entity", sort=False):
        _detect_recon(group, rows)
        _detect_cred_attack(group, rows)
        _detect_compromise(group, rows)
        _detect_priv_esc(group, rows)
        _detect_persistence(group, rows)
        _detect_exfil(group, rows)

    if not rows:
        return _empty_signals()

    result = pd.DataFrame([s.__dict__ for s in rows], columns=STAGE_SIGNAL_COLUMNS)
    result["ts"] = pd.to_datetime(result["ts"], errors="coerce")
    result["eid"] = result["eid"].astype(int)
    result = result.sort_values(["entity", "ts", "eid"], kind="stable").reset_index(drop=True)
    return result


def _progression_for_day(group: pd.DataFrame) -> dict:
    ordered = group.sort_values(["ts", "eid"], kind="stable")

    accepted: list[pd.Series] = []
    last_index = -1
    invalidated = False

    for _, signal in ordered.iterrows():
        idx = STAGE_INDEX[str(signal["stage"])]

        if idx < last_index:
            # A lower stage appearing after a later stage makes the later
            # continuation unreliable. Keep the valid prefix only.
            invalidated = True
            break

        accepted.append(signal)
        last_index = idx

    if not accepted:
        return {
            "entity": str(group["entity"].iloc[0]),
            "day": pd.Timestamp(group["ts"].min()).date().isoformat(),
            "stages_detected": "",
            "stage_times": "",
            "stages_completed": 0,
            "current_stage": "",
            "next_expected_stage": "",
            "progression_score": 0.0,
            "first_warning_ts": "",
        }

    stages = [str(x["stage"]) for x in accepted]
    times = [pd.Timestamp(x["ts"]) for x in accepted]
    distinct = list(dict.fromkeys(stages))
    current = max(distinct, key=lambda stage: STAGE_INDEX[stage])
    current_idx = STAGE_INDEX[current]
    next_stage = STAGES[current_idx + 1] if current_idx + 1 < len(STAGES) else ""

    # The score measures canonical stages reached, not raw signal count.
    score = round(100.0 * len(distinct) / len(STAGES), 2)

    if invalidated:
        # Do not claim a complete progression when chronological evidence
        # contains a lower stage after a later one.
        score = min(score, 100.0 * len(distinct) / len(STAGES))

    return {
        "entity": str(group["entity"].iloc[0]),
        "day": pd.Timestamp(group["ts"].min()).date().isoformat(),
        "stages_detected": "|".join(stages),
        "stage_times": "|".join(t.isoformat() for t in times),
        "stages_completed": len(distinct),
        "current_stage": current,
        "next_expected_stage": next_stage,
        "progression_score": score,
        "first_warning_ts": times[0].isoformat(),
    }


def evaluate_progression(
    stage_signals: pd.DataFrame,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Build one progression row per entity-day from stage signals."""
    _ = cfg or Config()

    if stage_signals is None or stage_signals.empty:
        return _empty_progression()

    required = set(STAGE_SIGNAL_COLUMNS)
    missing = sorted(required - set(stage_signals.columns))
    if missing:
        raise ValueError(f"stage_signals missing required columns: {missing}")

    frame = stage_signals.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    if frame["ts"].isna().any():
        raise ValueError("stage_signals contains invalid timestamps")
    frame["eid"] = pd.to_numeric(frame["eid"], errors="raise").astype(int)
    frame["stage"] = frame["stage"].astype(str)
    frame["entity"] = frame["entity"].astype(str)

    rows = []
    frame = frame.sort_values(["entity", "ts", "eid"], kind="stable")
    frame["day"] = frame["ts"].dt.date

    for (_, _), group in frame.groupby(["entity", "day"], sort=False):
        rows.append(_progression_for_day(group))

    if not rows:
        return _empty_progression()

    return pd.DataFrame(rows, columns=PROGRESSION_COLUMNS)
