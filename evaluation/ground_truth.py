"""Load and validate ground_truth.csv. Used only by the evaluation harness."""
from __future__ import annotations

from ipaddress import ip_address, ip_network
from pathlib import Path

import pandas as pd

from intruder.config import Config
from intruder.contracts import SPLITS, STAGE_INDEX, ContractError, _raise, _require, _split, _text, _times
from intruder.events import SYNTHETIC_EXTERNAL

GT_COLUMNS = ["split", "scenario_id", "entity_id", "day", "label", "trap_type", "attack_start", "attack_end",
              "expected_stages", "profile"]
LABELS = ("benign", "benign_trap", "failed_attack", "successful_intrusion")
ATTACK_LABELS = ("failed_attack", "successful_intrusion")
TRAP_TYPES = ("typo_login", "internal_backup", "failed_external_auth", "noisy_benign")
PROFILES = ("base", "shifted")
PRE_COMPROMISE = {"RECON", "CRED_ATTACK"}


def _synthetic_ip(value: str) -> bool:
    try:
        ip = ip_address(value)
    except ValueError:
        return False
    internal = (ip_network(n) for n in Config().internal_nets)
    return any(ip in n for n in internal) or any(ip in n for n in SYNTHETIC_EXTERNAL)


def validate_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised copy (day -> date, attack_* -> naive UTC timestamps) or raise ContractError."""
    _require(df, GT_COLUMNS, "ground truth")
    out = df.copy()
    for col in ("split", "scenario_id", "entity_id", "label", "trap_type", "expected_stages", "profile"):
        out[col] = _text(out[col])
    problems = []
    if bad := sorted(set(out.split) - set(SPLITS)):
        problems.append(f"unknown splits {bad}")
    if bad := sorted(set(out.label) - set(LABELS)):
        problems.append(f"unknown labels {bad}")
    if bad := sorted(set(out.profile) - set(PROFILES)):
        problems.append(f"unknown profiles {bad}")
    if bad := sorted(set(out.entity_id[~out.entity_id.map(_synthetic_ip)])):
        problems.append(f"entity_id must be a synthetic IP: {bad[:5]}")
    if (out.scenario_id == "").any():
        problems.append("scenario_id must not be empty")
    if out.duplicated(["split", "entity_id", "day"]).any():
        problems.append("duplicate (split, entity_id, day) rows")
    if ((out.split == "train") & out.label.isin(ATTACK_LABELS)).any():
        problems.append("train split must contain benign / benign_trap rows only")

    day = pd.to_datetime(_text(out.day), errors="coerce", format="ISO8601")
    start, bad_start = _times(out.attack_start)
    end, bad_end = _times(out.attack_end)
    if day.isna().any():
        problems.append("day must be YYYY-MM-DD")
    if bad_start.any() or bad_end.any():
        problems.append("attack_start / attack_end must be ISO-8601")

    for i, row in enumerate(out.itertuples(index=False)):
        where = f"row {i} ({row.entity_id})"
        is_trap, is_attack = row.label == "benign_trap", row.label in ATTACK_LABELS
        if is_trap and row.trap_type not in TRAP_TYPES:
            problems.append(f"{where}: benign_trap needs trap_type in {TRAP_TYPES}")
        if not is_trap and row.trap_type:
            problems.append(f"{where}: trap_type is only allowed on benign_trap rows")
        stages = _split(row.expected_stages)
        if not is_attack:
            if pd.notna(start.iloc[i]) or pd.notna(end.iloc[i]) or stages:
                problems.append(f"{where}: attack_start/attack_end/expected_stages must be empty for {row.label}")
            continue
        if pd.isna(start.iloc[i]) or pd.isna(end.iloc[i]):
            problems.append(f"{where}: attack rows need attack_start and attack_end")
        elif end.iloc[i] < start.iloc[i]:
            problems.append(f"{where}: attack_end precedes attack_start")
        elif pd.notna(day.iloc[i]) and not (start.iloc[i].date() == end.iloc[i].date() == day.iloc[i].date()):
            problems.append(f"{where}: attack must start and end on its day (entity-day is the evaluation unit)")
        if unknown := [s for s in stages if s not in STAGE_INDEX]:
            problems.append(f"{where}: unknown expected_stages {unknown}")
            continue
        if not stages:
            problems.append(f"{where}: attack rows need expected_stages")
        elif [STAGE_INDEX[s] for s in stages] != sorted({STAGE_INDEX[s] for s in stages}):
            problems.append(f"{where}: expected_stages must be unique and in canonical stage order")
        if row.label == "failed_attack" and not set(stages) <= PRE_COMPROMISE:
            problems.append(f"{where}: failed_attack may only expect {sorted(PRE_COMPROMISE)}")
        if row.label == "successful_intrusion" and "COMPROMISE" not in stages:
            problems.append(f"{where}: successful_intrusion must expect COMPROMISE")
    _raise("ground truth", problems)
    out["day"], out["attack_start"], out["attack_end"] = day.dt.date, start, end
    return out


def load_ground_truth(paths) -> pd.DataFrame:
    """Load one or more ground_truth.csv files (e.g. one per split folder) and validate them together."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
    if not frames:
        raise ContractError("no ground truth files given")
    return validate_ground_truth(pd.concat(frames, ignore_index=True))
