"""Tunable settings. Detection thresholds that can be calibrated on validation data are marked as such."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Networks considered "inside" the organisation. Anything else is treated as external.
    internal_nets: tuple[str, ...] = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    # Windows starting in [start, end) hours are business hours; others get off_hours = 1.
    business_hours: tuple[int, int] = (7, 20)
    window_min: int = 5

    # Actor attribution for host-local events (see features.attribute_actors).
    session_hours: float = 8.0
    account_link_min: float = 60.0

    # host name -> ip, from hosts.csv (a plain inventory, never labels)
    inventory: dict[str, str] = field(default_factory=dict)
