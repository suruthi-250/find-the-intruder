"""Regression tests for the Phase 7.1 ground-truth correction (attacker IP on successful intrusions)."""
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from evaluation.ground_truth import GT_COLUMNS, load_ground_truth, validate_ground_truth
from generator.generate import SCENARIO_MIX, generate_dataset, scenario_to_ground_truth
from generator.scenarios import HOSTS, build_scenario

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "generated"
DAY = datetime(2026, 10, 1, tzinfo=timezone.utc)
HOST_IPS = {ip for _, ip in HOSTS}


def test_successful_intrusion_lists_host_and_attacker():
    result = build_scenario("successful_intrusion", random.Random(3), DAY, "t-intrusion")
    rows = scenario_to_ground_truth(result, "val")
    assert [r["entity_id"] for r in rows] == [result.entity_id, result.attacker_ip]
    host, attacker = rows
    assert host["entity_id"] in HOST_IPS and attacker["entity_id"] not in HOST_IPS
    assert attacker["label"] == host["label"] == "successful_intrusion"
    assert (attacker["attack_start"], attacker["attack_end"]) == (host["attack_start"], host["attack_end"])
    assert "COMPROMISE" in attacker["expected_stages"].split("|")

    events = pd.DataFrame(result.events)
    from_attacker = events[events.source_ip == result.attacker_ip]
    assert {"FAILED_LOGIN", "SUCCESS_LOGIN"} <= set(from_attacker.event_type)
    validate_ground_truth(pd.DataFrame(rows, columns=GT_COLUMNS))


@pytest.mark.parametrize("scenario", [s for s in SCENARIO_MIX if s != "successful_intrusion"])
def test_other_scenarios_keep_a_single_row(scenario):
    result = build_scenario(scenario, random.Random(3), DAY, f"t-{scenario}")
    assert len(scenario_to_ground_truth(result, "val")) == 1
    assert result.attacker_ip == ""


@pytest.mark.skipif(not DATA.exists(), reason="generated dataset not present")
@pytest.mark.parametrize("split", ["val", "test_id", "test_ood"])
def test_generated_ground_truth_labels_the_attacker_ip(split):
    gt = load_ground_truth(DATA / split / "ground_truth.csv")
    events = pd.read_csv(DATA / split / "events.csv", dtype=str, keep_default_na=False)
    events["day"] = pd.to_datetime(events.timestamp).dt.date
    intrusions = gt[gt.label == "successful_intrusion"]
    assert intrusions.scenario_id.nunique() == 8
    for _, rows in intrusions.groupby("scenario_id"):
        assert len(rows) == 2
        host = rows[rows.entity_id.isin(HOST_IPS)]
        attacker = rows[~rows.entity_id.isin(HOST_IPS)]
        assert len(host) == 1 and len(attacker) == 1
        r = attacker.iloc[0]
        day = events[events.day == r.day]
        external_sources = set(day.source_ip[(day.source_ip != "") & ~day.source_ip.str.startswith("10.")])
        assert external_sources == {r.entity_id}                    # the only external actor that day
        logins = day[(day.source_ip == r.entity_id) & (day.event_type == "SUCCESS_LOGIN")]
        assert len(logins) >= 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not DATA.exists(), reason="generated dataset not present")
def test_regenerating_with_seed_42_is_byte_identical(tmp_path):
    generate_dataset(tmp_path, seed=42)
    committed = sorted(p.relative_to(DATA) for p in DATA.rglob("*.csv"))
    regenerated = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.csv"))
    assert committed == regenerated
    mismatched = [str(p) for p in committed if sha256(DATA / p) != sha256(tmp_path / p)]
    assert mismatched == []
