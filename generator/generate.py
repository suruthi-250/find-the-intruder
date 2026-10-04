from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from generator.scenarios import (
    EVENT_FIELDS,
    HOSTS,
    SCENARIO_BUILDERS,
    ScenarioResult,
    build_scenario,
)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test_id", "test_ood")

SCENARIO_MIX = (
    "benign",
    "typo_login",
    "internal_backup",
    "failed_external_auth",
    "noisy_benign",
    "failed_attack",
    "successful_intrusion",
)


# Each split gets a collection of independent scenario-days.
#
# Evaluation splits cycle through SCENARIO_MIX, so 56 days give 8 instances of
# every scenario type. OOD uses the "shifted" behavioural profile (see
# generator.scenarios.PROFILE_PARAMS) and its own random seeds.
SPLIT_COUNTS = {
    "train": 24,
    "val": 56,
    "test_id": 56,
    "test_ood": 56,
}


# ---------------------------------------------------------------------------
# Ground-truth schema
# ---------------------------------------------------------------------------

GROUND_TRUTH_FIELDS = [
    "split",
    "scenario_id",
    "entity_id",
    "day",
    "label",
    "trap_type",
    "attack_start",
    "attack_end",
    "expected_stages",
    "profile",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic intrusion-detection datasets."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/generated"),
        help="Output directory.",
    )

    return parser.parse_args()


def utc_day(day_number: int) -> datetime:
    """Return a deterministic UTC day."""

    base = datetime(
        2026,
        10,
        1,
        tzinfo=timezone.utc,
    )

    return base + timedelta(days=day_number)


def scenario_types_for_split(
    split: str,
    index: int,
) -> list[str]:
    """
    Select scenarios deterministically.

    Train contains benign activity and benign traps only.
    Validation and test splits contain benign activity, traps,
    failed attacks, and successful intrusions.
    """

    # Training must not contain attack scenarios: one of each trap, then benign.
    if split == "train":
        if index < 5:
            return [SCENARIO_MIX[index]]

        return ["benign"]

    # Validation and test splits cycle through every scenario type.
    return [SCENARIO_MIX[index % len(SCENARIO_MIX)]]


def profile_for_split(split: str) -> str:
    if split == "test_ood":
        return "shifted"

    return "base"


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def normalise_event(
    event: dict[str, Any],
    eid: int,
) -> dict[str, Any]:

    row = dict(event)

    row["eid"] = eid

    # Keep empty optional fields as empty CSV cells.
    for field in (
        "source_ip",
        "destination_ip",
        "host",
        "username",
        "request_marker",
        "privilege",
    ):
        if row.get(field) is None:
            row[field] = ""

    return row


def scenario_to_ground_truth(
    result: ScenarioResult,
    split: str,
) -> list[dict[str, Any]]:
    """Ground-truth rows for one scenario-day.

    A successful intrusion yields two evaluation entities: the compromised
    host (as before) and the attacker's own source IP, on which the stages up
    to persistence actually occur.
    """

    row = {
        "split": split,
        "scenario_id": result.scenario_id,
        "entity_id": result.entity_id,
        "day": result.day,
        "label": result.label,
        "trap_type": result.trap_type,
        "attack_start": result.attack_start,
        "attack_end": result.attack_end,
        "expected_stages": result.expected_stages,
        "profile": result.profile,
    }

    rows = [row]

    if result.attacker_ip:
        rows.append({
            **row,
            "entity_id": result.attacker_ip,
            "expected_stages": result.attacker_stages,
        })

    return rows


# ---------------------------------------------------------------------------
# Hosts
# ---------------------------------------------------------------------------

def collect_hosts() -> list[dict[str, str]]:
    """The canonical host inventory. Never inferred from event traffic."""

    return [
        {
            "host": host,
            "ip": ip,
        }
        for host, ip in sorted(HOSTS)
    ]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_split(
    split: str,
    count: int,
    rng: random.Random,
    global_day_offset: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    all_events: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []

    profile = profile_for_split(split)

    eid = 1

    for index in range(count):

        scenario_types = scenario_types_for_split(
            split,
            index,
        )

        day_start = utc_day(
            global_day_offset + index
        )

        for scenario_number, scenario_type in enumerate(
            scenario_types
        ):

            scenario_id = (
                f"{split}-{index:03d}-{scenario_number:02d}-"
                f"{scenario_type}"
            )

            # Use a fresh deterministic RNG for every scenario.
            scenario_seed = rng.randint(
                0,
                2**32 - 1,
            )

            scenario_rng = random.Random(
                scenario_seed
            )

            result = build_scenario(
                scenario_type,
                scenario_rng,
                day_start,
                scenario_id,
                profile=profile,
            )

            for event in result.events:
                all_events.append(
                    normalise_event(
                        event,
                        eid,
                    )
                )

                eid += 1

            if result.label != "benign":
                ground_truth.extend(
                    scenario_to_ground_truth(
                        result,
                        split,
                    )
                )

    return all_events, ground_truth


def generate_dataset(
    output_dir: Path,
    seed: int,
) -> None:

    rng = random.Random(seed)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary: list[tuple[str, int, int]] = []

    day_offset = 0

    for split in SPLITS:

        events, ground_truth = generate_split(
            split,
            SPLIT_COUNTS[split],
            rng,
            day_offset,
        )

        day_offset += SPLIT_COUNTS[split]

        split_dir = output_dir / split
        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        events_path = split_dir / "events.csv"
        ground_truth_path = split_dir / "ground_truth.csv"

        event_fields = [
            "eid",
            *EVENT_FIELDS,
        ]

        write_csv(
            events_path,
            events,
            event_fields,
        )

        write_csv(
            ground_truth_path,
            ground_truth,
            GROUND_TRUTH_FIELDS,
        )

        summary.append(
            (
                split,
                len(events),
                len(ground_truth),
            )
        )

    hosts_rows = collect_hosts()

    write_csv(
        output_dir / "hosts.csv",
        hosts_rows,
        ["host", "ip"],
    )

    print()
    print("Synthetic dataset generated.")
    print()
    print(f"Output: {output_dir}")
    print()
    print(
        f"{'split':12} {'events':>10} {'ground_truth':>15}"
    )
    print("-" * 40)

    for split, event_count, truth_count in summary:
        print(
            f"{split:12} "
            f"{event_count:10} "
            f"{truth_count:15}"
        )

    print()
    print(f"hosts: {len(hosts_rows)}")
    print(f"seed:  {seed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    generate_dataset(
        output_dir=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()