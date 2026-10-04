"""Score splits with the trained detector and write A/B/C predictions.

    py predict.py [--data data/generated] [--model models/isolation_forest.pkl]
                  [--splits val test_id test_ood] [--out results]

Per split: events -> actor attribution -> 5-minute features -> anomaly scores, and
events -> stage signals -> temporal progression; both feed the three detectors.

Writes <out>/predictions.csv (the input to `py -m evaluation.run`) and, per split,
<out>/<split>/window_scores.csv, stage_signals.csv and progression.csv. Only events.csv and
hosts.csv are read; every intermediate frame is checked against intruder.contracts.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from intruder.config import Config
from intruder.contracts import (DETECTORS, validate_predictions, validate_progression, validate_stage_signals,
                                validate_window_scores)
from intruder.features import attribute_actors, build_features
from intruder.model import IsolationForestDetector
from intruder.prediction import build_all_detectors
from intruder.rules import detect_stage_signals, evaluate_progression
from train import MODEL_FILE, load_config, load_split

DEFAULT_SPLITS = ("val", "test_id", "test_ood")
PREDICTIONS_FILE = "predictions.csv"


def load_detector(path: Path) -> IsolationForestDetector:
    with path.open("rb") as f:
        detector = pickle.load(f)
    if not isinstance(detector, IsolationForestDetector) or not detector.is_fitted:
        raise ValueError(f"{path} does not contain a fitted IsolationForestDetector")
    return detector


def predict_split(events: pd.DataFrame, cfg: Config, detector: IsolationForestDetector,
                  split: str) -> dict[str, pd.DataFrame]:
    """Run the detection pipeline on one split's events."""
    attributed = attribute_actors(events, cfg)
    features = build_features(events, cfg)
    window_scores = validate_window_scores(detector.score_windows(features))

    signals = detect_stage_signals(attributed, cfg)
    if not signals.empty:
        validate_stage_signals(signals)
    progression = evaluate_progression(signals, cfg)
    if not progression.empty:
        validate_progression(progression)

    predictions = build_all_detectors(window_scores, progression, split=split, stage_signals=signals)
    return {"features": features, "window_scores": window_scores, "stage_signals": signals,
            "progression": progression, "predictions": predictions}


def predict(data: Path, model: Path, splits, out: Path) -> pd.DataFrame:
    cfg = load_config(data)
    detector = load_detector(model)
    all_predictions = []
    for split in splits:
        events, report = load_split(data, split, cfg)
        result = predict_split(events, cfg, detector, split)

        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in ("window_scores", "stage_signals", "progression"):
            result[name].to_csv(split_dir / f"{name}.csv", index=False)

        preds = result["predictions"]
        all_predictions.append(preds)
        alerts = preds[preds.alert].groupby("detector").size().reindex(list(DETECTORS), fill_value=0)
        entity_days = preds[preds.detector == DETECTORS[0]].shape[0]
        print(f"{split:9} events={report.valid} (skipped {report.skipped}) windows={len(result['features'])} "
              f"entity-days={entity_days} stage signals={len(result['stage_signals'])} "
              f"alerts: " + ", ".join(f"{d}={n}" for d, n in alerts.items()))

    predictions = validate_predictions(pd.concat(all_predictions, ignore_index=True))
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / PREDICTIONS_FILE, index=False)
    print(f"predictions: {out / PREDICTIONS_FILE} ({len(predictions)} rows)")
    return predictions


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--model", type=Path, default=Path("models") / MODEL_FILE)
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    return predict(args.data, args.model, args.splits, args.out)


if __name__ == "__main__":
    main()
