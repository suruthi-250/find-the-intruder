"""Structural checks on the synthetic generator: split composition, schema, determinism and trap timing."""
import random
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import pytest

from generator.generate import SCENARIO_MIX, SPLIT_COUNTS, collect_hosts, scenario_types_for_split
from generator.scenarios import EVENT_FIELDS, HOSTS, PROFILE_PARAMS, add_benign_background, build_scenario

DAY = datetime(2026, 10, 1, tzinfo=timezone.utc)
ATTACKS = {"failed_attack", "successful_intrusion"}
TRAPS = ("typo_login", "internal_backup", "failed_external_auth", "noisy_benign")


def split_mix(split):
    return Counter(t for i in range(SPLIT_COUNTS[split]) for t in scenario_types_for_split(split, i))


def frame(scenario, seed, profile):
    result = build_scenario(scenario, random.Random(seed), DAY, f"t-{scenario}", profile=profile)
    df = pd.DataFrame(result.events)
    df["ts"] = pd.to_datetime(df.timestamp, utc=True)
    return result, df


def trap_events(trap, seed, profile):
    """Only the trap's own events: traps append them after the seeded background."""
    background = []
    add_benign_background(background, DAY, random.Random(seed), profile=profile)
    result, df = frame(trap, seed, profile)
    return result, df.iloc[len(background):]


def test_train_has_no_attacks_and_every_trap():
    mix = split_mix("train")
    assert not ATTACKS & set(mix)
    assert all(mix[t] >= 1 for t in TRAPS)


@pytest.mark.parametrize("split", ["val", "test_id", "test_ood"])
def test_evaluation_splits_are_balanced(split):
    assert split_mix(split) == {t: 8 for t in SCENARIO_MIX}


def test_host_inventory_is_canonical():
    assert collect_hosts() == [{"host": h, "ip": ip} for h, ip in sorted(HOSTS)]
    assert len(collect_hosts()) == 3


def test_profiles_define_the_same_parameters():
    assert PROFILE_PARAMS["base"].keys() == PROFILE_PARAMS["shifted"].keys()


@pytest.mark.parametrize("profile", ["base", "shifted"])
@pytest.mark.parametrize("scenario", SCENARIO_MIX)
def test_events_use_only_the_event_schema(scenario, profile):
    result, _ = frame(scenario, 1, profile)
    assert all(list(e) == EVENT_FIELDS for e in result.events)


@pytest.mark.parametrize("profile", ["base", "shifted"])
def test_generation_is_deterministic(profile):
    a, _ = frame("internal_backup", 7, profile)
    b, _ = frame("internal_backup", 7, profile)
    assert a.events == b.events


def test_shifted_background_is_denser():
    base = sum(len(frame("benign", s, "base")[0].events) for s in range(3))
    shifted = sum(len(frame("benign", s, "shifted")[0].events) for s in range(3))
    assert shifted > 1.15 * base


@pytest.mark.parametrize("profile", ["base", "shifted"])
@pytest.mark.parametrize("seed", range(15))
def test_traps_stay_in_their_day_and_daytime_traps_stay_in_business_hours(profile, seed):
    for trap in TRAPS:
        _, own = trap_events(trap, seed, profile)
        assert len(own) > 0, trap
        assert (own.ts.dt.date == DAY.date()).all(), trap
        if trap in ("typo_login", "noisy_benign"):
            assert own.ts.dt.hour.between(7, 19).all(), trap   # never flips off_hours


@pytest.mark.parametrize("profile", ["base", "shifted"])
@pytest.mark.parametrize("seed", range(15))
def test_slow_failed_auth_stays_low_rate(profile, seed):
    _, own = trap_events("failed_external_auth", seed, profile)
    assert (own.event_type == "FAILED_LOGIN").all()
    assert own.groupby(own.ts.dt.floor("5min")).size().max() <= 3
