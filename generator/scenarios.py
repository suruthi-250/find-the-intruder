from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import random
from typing import Any


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

EVENT_FIELDS = [
    "timestamp",
    "source_ip",
    "destination_ip",
    "host",
    "username",
    "event_type",
    "status",
    "port",
    "bytes",
    "http_status",
    "request_marker",
    "privilege",
]


@dataclass
class ScenarioResult:
    """Events plus metadata used later to create ground_truth.csv."""

    scenario_id: str
    day: str
    events: list[dict[str, Any]]
    label: str
    trap_type: str = ""
    entity_id: str = ""
    attack_start: str = ""
    attack_end: str = ""
    expected_stages: str = ""
    profile: str = "base"
    # Successful intrusions have a second evaluation entity on the same day: the
    # attacker's own source IP and the stages carried out from it. Metadata only.
    attacker_ip: str = ""
    attacker_stages: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERNAL_NETWORK = "10.0.0."
EXTERNAL_NETWORKS = (
    "192.0.2.",
    "198.51.100.",
    "203.0.113.",
)

HOSTS = (
    ("web-01", "10.0.0.10"),
    ("app-01", "10.0.0.20"),
    ("db-01", "10.0.0.30"),
)

USERS = ("alice", "bob", "charlie", "david")
COMMON_PORTS = (22, 53, 80, 443, 3389)
NOISY_PORTS = (22, 53, 80, 443, 445, 3389, 8080, 8443, 21, 25, 3306, 5432)
NOISY_ALLOWED_PORTS = (22, 80, 443, 3389)

# Backup and slow-auth traps may start at any hour in both profiles. Ending at
# 22:59 keeps every trap inside its own day (the evaluation unit is entity-day).
TRAP_ANY_HOUR = (0, 22)


# ---------------------------------------------------------------------------
# Behavioural profiles
#
# "shifted" changes only rates, sizes and timing ranges of behaviour that also
# exists under "base"; no event type, field or value is unique to a profile.
# Benign-background values for "base" are the original constants, so base
# background is generated exactly as before. Trap ranges are sampled per
# scenario in both profiles; shifted ranges overlap base but are wider.
# Attack scenarios are unchanged and inherit the profile's benign background.
# ---------------------------------------------------------------------------

PROFILE_PARAMS: dict[str, dict[str, Any]] = {
    "base": {
        # benign background
        "client_pool": (40, 220),
        "event_prob": 0.32,
        "event_weights": (20, 30, 25, 15, 10),  # login, web, connection, transfer, sudo
        "login_fail_prob": 0.15,
        "deny_weights": (85, 15),               # ALLOW, DENY
        "web_status_weights": (70, 8, 5, 17),   # 200, 204, 301, 404
        "internal_bytes": (5_000, 80_000),
        "sudo_weights": (90, 10),               # ROUTINE, ELEVATED_SHELL
        "admin_account_prob": 0.08,
        "external_prob": 0.10,
        "external_bytes": (1_000, 20_000),
        # trap A - typo login (daytime)
        "typo_hours": (8, 16),
        "typo_failures": (2, 4),
        "typo_gap_s": (10, 30),
        "typo_success_after_s": (20, 60),
        # trap B - internal backup
        "backup_transfers": (8, 14),
        "backup_gap_s": (90, 180),
        "backup_bytes": (400_000, 2_200_000),
        # trap C - slow failed external auth
        "slow_auth_failures": (4, 8),
        "slow_auth_gap_s": (150, 240),
        # trap D - noisy admin workstation (daytime)
        "noisy_hours": (9, 16),
        "noisy_ports": (6, 9),
        "noisy_gap_s": (8, 16),
        "noisy_404s": (3, 6),
        "noisy_404_gap_s": (10, 20),
    },
    "shifted": {
        "client_pool": (40, 130),
        "event_prob": 0.42,
        "event_weights": (16, 36, 22, 16, 10),
        "login_fail_prob": 0.22,
        "deny_weights": (78, 22),
        "web_status_weights": (63, 8, 5, 24),
        "internal_bytes": (10_000, 150_000),
        "sudo_weights": (85, 15),
        "admin_account_prob": 0.12,
        "external_prob": 0.15,
        "external_bytes": (2_000, 40_000),
        "typo_hours": (8, 18),
        "typo_failures": (3, 6),
        "typo_gap_s": (8, 40),
        "typo_success_after_s": (15, 90),
        "backup_transfers": (10, 18),
        "backup_gap_s": (75, 180),
        "backup_bytes": (500_000, 2_600_000),
        "slow_auth_failures": (5, 10),
        "slow_auth_gap_s": (105, 210),   # still at most 3 failures per 5-minute window
        "noisy_hours": (8, 18),
        "noisy_ports": (8, 11),
        "noisy_gap_s": (6, 18),
        "noisy_404s": (4, 9),
        "noisy_404_gap_s": (8, 24),
    },
}


def iso(ts: datetime) -> str:
    """Return an ISO-8601 UTC timestamp."""

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_event(
    ts: datetime,
    *,
    source_ip: str | None = None,
    destination_ip: str | None = None,
    host: str | None = None,
    username: str | None = None,
    event_type: str,
    status: str,
    port: int | None = None,
    bytes_: int | None = None,
    http_status: int | None = None,
    request_marker: str | None = None,
    privilege: str | None = None,
) -> dict[str, Any]:
    """Create one event using the project's event contract."""

    return {
        "timestamp": iso(ts),
        "source_ip": source_ip or "",
        "destination_ip": destination_ip or "",
        "host": host or "",
        "username": username or "",
        "event_type": event_type,
        "status": status,
        "port": port,
        "bytes": bytes_,
        "http_status": http_status,
        "request_marker": request_marker,
        "privilege": privilege,
    }


def random_internal_ip(
    rng: random.Random,
    pool: tuple[int, int] = (40, 220),
) -> str:
    return f"{INTERNAL_NETWORK}{rng.randint(*pool)}"


def random_external_ip(rng: random.Random) -> str:
    network = rng.choice(EXTERNAL_NETWORKS)
    return f"{network}{rng.randint(1, 220)}"


def choose_host(rng: random.Random) -> tuple[str, str]:
    return rng.choice(HOSTS)


def start_time(
    rng: random.Random,
    day_start: datetime,
    hours: tuple[int, int],
) -> datetime:
    """Random minute within the inclusive hour range."""

    return day_start + timedelta(
        hours=rng.randint(*hours),
        minutes=rng.randint(0, 59),
    )


def normal_web_event(
    rng: random.Random,
    ts: datetime,
    source_ip: str,
    host: str,
    status_weights: tuple[int, ...] = (70, 8, 5, 17),
) -> dict[str, Any]:
    """Normal web traffic."""

    status_code = rng.choices(
        [200, 204, 301, 404],
        weights=status_weights,
        k=1,
    )[0]

    return make_event(
        ts,
        source_ip=source_ip,
        destination_ip=next(ip for name, ip in HOSTS if name == host),
        host=host,
        event_type="WEB_REQUEST",
        status="SUCCESS",
        port=rng.choice((80, 443)),
        http_status=status_code,
        request_marker="NONE",
    )


def add_benign_background(
    events: list[dict[str, Any]],
    start: datetime,
    rng: random.Random,
    *,
    profile: str = "base",
) -> None:
    """
    Add ordinary activity.

    Importantly, benign traffic uses the same event types that attack
    scenarios can use. This prevents trivial shortcuts such as
    'SUDO_EVENT always means attack'.
    """

    p = PROFILE_PARAMS[profile]

    for minute in range(24 * 60):
        ts = start + timedelta(minutes=minute)

        # Most minutes have no event.
        if rng.random() > p["event_prob"]:
            continue

        host, host_ip = choose_host(rng)
        internal_user_ip = random_internal_ip(rng, p["client_pool"])

        event_choice = rng.choices(
            [
                "login",
                "web",
                "connection",
                "transfer",
                "sudo",
            ],
            weights=p["event_weights"],
            k=1,
        )[0]

        if event_choice == "login":
            username = rng.choice(USERS)

            # Mostly successful logins, occasional ordinary failures.
            if rng.random() < p["login_fail_prob"]:
                events.append(
                    make_event(
                        ts,
                        source_ip=internal_user_ip,
                        host=host,
                        username=username,
                        event_type="FAILED_LOGIN",
                        status="FAILURE",
                    )
                )
            else:
                events.append(
                    make_event(
                        ts,
                        source_ip=internal_user_ip,
                        host=host,
                        username=username,
                        event_type="SUCCESS_LOGIN",
                        status="SUCCESS",
                    )
                )

        elif event_choice == "web":
            events.append(
                normal_web_event(
                    rng,
                    ts,
                    internal_user_ip,
                    host,
                    p["web_status_weights"],
                )
            )

        elif event_choice == "connection":
            port = rng.choice(COMMON_PORTS)

            events.append(
                make_event(
                    ts,
                    source_ip=internal_user_ip,
                    destination_ip=host_ip,
                    host=host,
                    event_type="CONNECTION",
                    status=rng.choices(
                        ["ALLOW", "DENY"],
                        weights=p["deny_weights"],
                        k=1,
                    )[0],
                    port=port,
                )
            )

        elif event_choice == "transfer":
            destination = random_internal_ip(rng)

            events.append(
                make_event(
                    ts,
                    source_ip=host_ip,
                    destination_ip=destination,
                    host=host,
                    event_type="INTERNAL_TRANSFER",
                    status="ALLOW",
                    bytes_=rng.randint(*p["internal_bytes"]),
                )
            )

        elif event_choice == "sudo":
            username = rng.choice(USERS)

            events.append(
                make_event(
                    ts,
                    source_ip="",
                    host=host,
                    username=username,
                    event_type="SUDO_EVENT",
                    status="SUCCESS",
                    privilege=rng.choices(
                        ["ROUTINE", "ELEVATED_SHELL"],
                        weights=p["sudo_weights"],
                        k=1,
                    )[0],
                )
            )

            # Occasionally a legitimate administrator creates an account.
            # This prevents NEW_ACCOUNT from becoming an attack-only signal.
            if rng.random() < p["admin_account_prob"]:
                events.append(
                    make_event(
                        ts + timedelta(seconds=20),
                        source_ip="",
                        host=host,
                        username="admin_" + str(rng.randint(10, 99)),
                        event_type="NEW_ACCOUNT",
                        status="SUCCESS",
                    )
                )

            # Occasionally a legitimate host communicates externally.
            # Keep the volume relatively small so it is not equivalent to
            # the synthetic intrusion's large outbound transfer.
            if rng.random() < p["external_prob"]:
                events.append(
                    make_event(
                        ts + timedelta(seconds=40),
                        source_ip=host_ip,
                        destination_ip=random_external_ip(rng),
                        host=host,
                        event_type="EXTERNAL_TRANSFER",
                        status="ALLOW",
                        bytes_=rng.randint(*p["external_bytes"]),
                    )
                )

# ---------------------------------------------------------------------------
# Benign scenario
# ---------------------------------------------------------------------------

def benign_scenario(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="benign",
        entity_id="",
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Trap A — small typo-login burst
# ---------------------------------------------------------------------------

def typo_login_trap(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    p = PROFILE_PARAMS[profile]

    host, _ = choose_host(rng)
    source_ip = random_internal_ip(rng, p["client_pool"])
    username = rng.choice(USERS)

    start = start_time(rng, day_start, p["typo_hours"])
    failures = rng.randint(*p["typo_failures"])
    gap = rng.randint(*p["typo_gap_s"])

    # Small burst of failed attempts, followed by a legitimate success.
    for i in range(failures):
        events.append(
            make_event(
                start + timedelta(seconds=i * gap),
                source_ip=source_ip,
                host=host,
                username=username,
                event_type="FAILED_LOGIN",
                status="FAILURE",
            )
        )

    events.append(
        make_event(
            start + timedelta(
                seconds=(failures - 1) * gap
                + rng.randint(*p["typo_success_after_s"])
            ),
            source_ip=source_ip,
            host=host,
            username=username,
            event_type="SUCCESS_LOGIN",
            status="SUCCESS",
        )
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="benign_trap",
        trap_type="typo_login",
        entity_id=source_ip,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Trap B — legitimate internal backup
# ---------------------------------------------------------------------------

def internal_backup_trap(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    p = PROFILE_PARAMS[profile]

    host, host_ip = choose_host(rng)
    backup_server = "10.0.0.250"

    start = start_time(rng, day_start, TRAP_ANY_HOUR)
    gap = rng.randint(*p["backup_gap_s"])

    for i in range(rng.randint(*p["backup_transfers"])):
        events.append(
            make_event(
                start + timedelta(seconds=i * gap),
                source_ip=host_ip,
                destination_ip=backup_server,
                host=host,
                event_type="INTERNAL_TRANSFER",
                status="ALLOW",
                bytes_=rng.randint(*p["backup_bytes"]),
            )
        )

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="benign_trap",
        trap_type="internal_backup",
        entity_id=host_ip,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Trap C — slow failed external authentication
# ---------------------------------------------------------------------------

def failed_external_auth_trap(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    p = PROFILE_PARAMS[profile]

    host, _ = choose_host(rng)
    attacker_like_ip = random_external_ip(rng)
    username = rng.choice(USERS)

    start = start_time(rng, day_start, TRAP_ANY_HOUR)
    gap = rng.randint(*p["slow_auth_gap_s"])

    # Deliberately slow and low-volume.
    for i in range(rng.randint(*p["slow_auth_failures"])):
        events.append(
            make_event(
                start + timedelta(seconds=i * gap),
                source_ip=attacker_like_ip,
                host=host,
                username=username,
                event_type="FAILED_LOGIN",
                status="FAILURE",
            )
        )

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="benign_trap",
        trap_type="failed_external_auth",
        entity_id=attacker_like_ip,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Trap D — noisy but legitimate activity
# ---------------------------------------------------------------------------

def noisy_benign_trap(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    p = PROFILE_PARAMS[profile]

    host, host_ip = choose_host(rng)
    source_ip = random_internal_ip(rng, p["client_pool"])

    start = start_time(rng, day_start, p["noisy_hours"])

    # Many connections to different ports, but all from an internal
    # administration workstation and without an attack progression.
    ports = rng.sample(NOISY_PORTS, rng.randint(*p["noisy_ports"]))
    gap = rng.randint(*p["noisy_gap_s"])

    for i, port in enumerate(ports):
        events.append(
            make_event(
                start + timedelta(seconds=i * gap),
                source_ip=source_ip,
                destination_ip=host_ip,
                host=host,
                event_type="CONNECTION",
                status="ALLOW" if port in NOISY_ALLOWED_PORTS else "DENY",
                port=port,
            )
        )

    # A few ordinary HTTP 4xx responses add another source of noise.
    web_start = start + timedelta(seconds=len(ports) * gap + 30)
    web_gap = rng.randint(*p["noisy_404_gap_s"])

    for i in range(rng.randint(*p["noisy_404s"])):
        events.append(
            make_event(
                web_start + timedelta(seconds=i * web_gap),
                source_ip=source_ip,
                destination_ip=host_ip,
                host=host,
                event_type="WEB_REQUEST",
                status="SUCCESS",
                port=443,
                http_status=404,
                request_marker="NONE",
            )
        )

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="benign_trap",
        trap_type="noisy_benign",
        entity_id=source_ip,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Failed attack — recon + credential attack, no compromise
# ---------------------------------------------------------------------------

def failed_attack_scenario(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    host, host_ip = choose_host(rng)
    attacker_ip = random_external_ip(rng)
    username = rng.choice(USERS)

    attack_start = day_start + timedelta(hours=13)

    # Stage 1: reconnaissance.
    recon_start = attack_start

    for i, port in enumerate(
        rng.sample(
            list(range(20, 9000)),
            14,
        )
    ):
        events.append(
            make_event(
                recon_start + timedelta(seconds=i * 4),
                source_ip=attacker_ip,
                destination_ip=host_ip,
                host=host,
                event_type="CONNECTION",
                status="DENY",
                port=port,
            )
        )

    # Stage 2: sustained credential attack.
    credential_start = attack_start + timedelta(minutes=3)

    for i in range(30):
        events.append(
            make_event(
                credential_start + timedelta(seconds=i * 5),
                source_ip=attacker_ip,
                host=host,
                username=username,
                event_type="FAILED_LOGIN",
                status="FAILURE",
            )
        )

    attack_end = credential_start + timedelta(minutes=3)

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="failed_attack",
        entity_id=attacker_ip,
        attack_start=iso(attack_start),
        attack_end=iso(attack_end),
        expected_stages="RECON|CRED_ATTACK",
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Successful intrusion
# ---------------------------------------------------------------------------

def successful_intrusion_scenario(
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:

    events: list[dict[str, Any]] = []

    add_benign_background(
        events,
        day_start,
        rng,
        profile=profile,
    )

    host, host_ip = choose_host(rng)
    attacker_ip = random_external_ip(rng)
    username = rng.choice(USERS)
    new_account = "service_" + str(rng.randint(10, 99))

    attack_start = day_start + timedelta(hours=18)

    # ------------------------------------------------------------------
    # RECON
    # ------------------------------------------------------------------

    for i, port in enumerate(
        rng.sample(
            list(range(20, 9000)),
            16,
        )
    ):
        events.append(
            make_event(
                attack_start + timedelta(seconds=i * 4),
                source_ip=attacker_ip,
                destination_ip=host_ip,
                host=host,
                event_type="CONNECTION",
                status="DENY",
                port=port,
            )
        )

    # ------------------------------------------------------------------
    # CREDENTIAL ATTACK
    # ------------------------------------------------------------------

    credential_start = attack_start + timedelta(minutes=3)

    for i in range(18):
        events.append(
            make_event(
                credential_start + timedelta(seconds=i * 5),
                source_ip=attacker_ip,
                host=host,
                username=username,
                event_type="FAILED_LOGIN",
                status="FAILURE",
            )
        )

    # ------------------------------------------------------------------
    # COMPROMISE
    # ------------------------------------------------------------------

    compromise_time = credential_start + timedelta(minutes=2)

    events.append(
        make_event(
            compromise_time,
            source_ip=attacker_ip,
            host=host,
            username=username,
            event_type="SUCCESS_LOGIN",
            status="SUCCESS",
        )
    )

    # ------------------------------------------------------------------
    # PRIVILEGE ESCALATION
    #
    # source_ip is intentionally empty here.
    # The feature attribution layer should associate this event with
    # the recent successful session.
    # ------------------------------------------------------------------

    privilege_time = compromise_time + timedelta(minutes=4)

    events.append(
        make_event(
            privilege_time,
            source_ip="",
            host=host,
            username=username,
            event_type="SUDO_EVENT",
            status="SUCCESS",
            privilege="ELEVATED_SHELL",
        )
    )

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    persistence_time = privilege_time + timedelta(minutes=3)

    events.append(
        make_event(
            persistence_time,
            source_ip="",
            host=host,
            username=new_account,
            event_type="NEW_ACCOUNT",
            status="SUCCESS",
        )
    )

    # ------------------------------------------------------------------
    # EXFILTRATION
    #
    # This originates from the compromised internal host, rather than
    # the external attacker IP. This forces the detection layer to
    # reason about progression across related entities.
    # ------------------------------------------------------------------

    exfil_time = persistence_time + timedelta(minutes=5)
    external_destination = random_external_ip(rng)

    events.append(
        make_event(
            exfil_time,
            source_ip=host_ip,
            destination_ip=external_destination,
            host=host,
            event_type="EXTERNAL_TRANSFER",
            status="ALLOW",
            bytes_=rng.randint(2_000_000, 5_000_000),
        )
    )

    attack_end = exfil_time + timedelta(minutes=2)

    return ScenarioResult(
        scenario_id=scenario_id,
        day=day_start.date().isoformat(),
        events=events,
        label="successful_intrusion",

        # For a compromised-host scenario, the evaluation entity is
        # allowed to be the compromised host IP.
        entity_id=host_ip,

        attack_start=iso(attack_start),
        attack_end=iso(attack_end),
        expected_stages=(
            "RECON|CRED_ATTACK|COMPROMISE|"
            "PRIV_ESC|PERSISTENCE|EXFIL"
        ),
        profile=profile,

        # The intrusion stages up to persistence act from the attacker's IP;
        # the outbound transfer originates from the compromised host above.
        attacker_ip=attacker_ip,
        attacker_stages="RECON|CRED_ATTACK|COMPROMISE|PRIV_ESC|PERSISTENCE",
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_BUILDERS = {
    "benign": benign_scenario,
    "typo_login": typo_login_trap,
    "internal_backup": internal_backup_trap,
    "failed_external_auth": failed_external_auth_trap,
    "noisy_benign": noisy_benign_trap,
    "failed_attack": failed_attack_scenario,
    "successful_intrusion": successful_intrusion_scenario,
}


def build_scenario(
    scenario_type: str,
    rng: random.Random,
    day_start: datetime,
    scenario_id: str,
    *,
    profile: str = "base",
) -> ScenarioResult:
    """Build one named scenario."""

    try:
        builder = SCENARIO_BUILDERS[scenario_type]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIO_BUILDERS))
        raise ValueError(
            f"Unknown scenario type {scenario_type!r}. "
            f"Valid types: {valid}"
        ) from exc

    return builder(
        rng,
        day_start,
        scenario_id,
        profile=profile,
    )


__all__ = [
    "EVENT_FIELDS",
    "ScenarioResult",
    "SCENARIO_BUILDERS",
    "build_scenario",
]