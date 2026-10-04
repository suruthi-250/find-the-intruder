"""Fit the Isolation Forest on the benign training split.

    py train.py [--data data/generated] [--out models] [--seed 42]

The generator builds `train` from benign activity and benign traps only, so every training window is used
and no labels are read. Writes the fitted detector and a JSON training summary to --out.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from intruder.config import Config
from intruder.events import LoadReport, load_events, load_hosts
from intruder.features import FEATURES, build_features
from intruder.model import IsolationForestDetector

MODEL_FILE = "isolation_forest.pkl"
SUMMARY_FILE = "training_summary.json"


def load_config(data: Path) -> Config:
    """Config with the host inventory from <data>/hosts.csv (if present)."""
    hosts = data / "hosts.csv"
    return Config(inventory=load_hosts(hosts) if hosts.exists() else {})


def load_split(data: Path, split: str, cfg: Config):
    """Validated events for one split: (events, LoadReport)."""
    return load_events(data / split / "events.csv", internal_nets=cfg.internal_nets)


def report_dict(report: LoadReport) -> dict:
    return {"rows": report.rows, "valid": report.valid, "skipped": report.skipped,
            "skip_reasons": dict(report.reasons)}


def train(data: Path, out: Path, seed: int = 42) -> dict:
    cfg = load_config(data)
    events, report = load_split(data, "train", cfg)
    features = build_features(events, cfg)
    detector = IsolationForestDetector(random_state=seed).fit(features)

    out.mkdir(parents=True, exist_ok=True)
    with (out / MODEL_FILE).open("wb") as f:
        pickle.dump(detector, f)

    summary = {
        "split": "train",
        "events": report_dict(report),
        "windows": int(len(features)),
        "entities": int(features.entity.nunique()),
        "days": int(features.day.nunique()),
        "features": list(FEATURES),
        "random_state": seed,
        "n_estimators": detector.n_estimators,
        "contamination": detector.contamination,
        "inventory": cfg.inventory,
    }
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42, help="Isolation Forest random_state")
    args = ap.parse_args(argv)

    summary = train(args.data, args.out, args.seed)
    print(f"trained on {summary['windows']} windows ({summary['entities']} entities, {summary['days']} days); "
          f"events valid={summary['events']['valid']} skipped={summary['events']['skipped']}")
    print(f"model:   {args.out / MODEL_FILE}")
    print(f"summary: {args.out / SUMMARY_FILE}")
    return summary


if __name__ == "__main__":
    main()
