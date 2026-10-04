"""End-to-end tests for train.py / predict.py on a small dataset built with the (unmodified) generator."""
import json
import random
import shutil
from pathlib import Path

import pandas as pd
import pytest

import predict
import train
from evaluation import run as evaluation_run
from generator.generate import GROUND_TRUTH_FIELDS, collect_hosts, generate_split, write_csv
from generator.scenarios import EVENT_FIELDS
from intruder.contracts import DETECTORS, validate_predictions, validate_progression, validate_stage_signals, \
    validate_window_scores
from intruder.features import FEATURES, build_features

ROOT = Path(__file__).parents[1]
SPLITS = ["val", "test_id"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    rng, offset = random.Random(11), 0
    for split, days in (("train", 5), ("val", 7), ("test_id", 7)):
        events, truth = generate_split(split, days, rng, offset)
        offset += days
        write_csv(data / split / "events.csv", events, ["eid", *EVENT_FIELDS])
        write_csv(data / split / "ground_truth.csv", truth, GROUND_TRUTH_FIELDS)
    write_csv(data / "hosts.csv", collect_hosts(), ["host", "ip"])
    return data


@pytest.fixture(scope="module")
def run(dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    summary = train.main(["--data", str(dataset), "--out", str(out / "models"), "--seed", "42"])
    predictions = predict.main(["--data", str(dataset), "--model", str(out / "models" / train.MODEL_FILE),
                                "--splits", *SPLITS, "--out", str(out / "results")])
    return {"out": out, "summary": summary, "predictions": predictions}


def test_training_writes_model_and_summary(run):
    models = run["out"] / "models"
    assert (models / train.MODEL_FILE).exists()
    summary = json.loads((models / train.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["windows"] > 0 and summary["events"]["skipped"] == 0
    assert summary["features"] == FEATURES
    assert summary["random_state"] == 42
    assert predict.load_detector(models / train.MODEL_FILE).is_fitted


def test_prediction_file_satisfies_the_contract(run):
    df = pd.read_csv(run["out"] / "results" / predict.PREDICTIONS_FILE, dtype=str, keep_default_na=False)
    out = validate_predictions(df)
    assert set(out.split) == set(SPLITS)
    assert set(out.detector) == set(DETECTORS)


def test_detectors_cover_identical_entity_days(run):
    preds = run["predictions"]
    for split in SPLITS:
        keys = {d: set(zip(g.entity, g.day)) for d, g in preds[preds.split == split].groupby("detector")}
        assert keys[DETECTORS[0]] == keys[DETECTORS[1]] == keys[DETECTORS[2]]


def test_every_processed_entity_day_is_predicted(run, dataset):
    cfg = train.load_config(dataset)
    for split in SPLITS:
        events, _ = train.load_split(dataset, split, cfg)
        features = build_features(events, cfg)
        expected = set(zip(features.entity, features.day))
        got = run["predictions"][(run["predictions"].split == split)
                                 & (run["predictions"].detector == "A_anomaly_only")]
        assert set(zip(got.entity, got.day)) == expected


def test_anomaly_only_has_no_stage_fields_and_alerts_have_warnings(run):
    preds = run["predictions"]
    a = preds[preds.detector == "A_anomaly_only"]
    assert (a.current_stage == "").all() and (a.stages_completed == 0).all()
    assert preds.loc[preds.alert, "first_warning_ts"].notna().all()
    assert preds.risk_score.between(0, 100).all()


def test_intermediate_outputs_are_written_and_valid(run):
    for split in SPLITS:
        d = run["out"] / "results" / split
        validate_window_scores(pd.read_csv(d / "window_scores.csv"))
        signals = pd.read_csv(d / "stage_signals.csv")
        if len(signals):
            validate_stage_signals(signals)
        progression = pd.read_csv(d / "progression.csv", dtype=str, keep_default_na=False)
        if len(progression):
            validate_progression(progression)


def test_pipeline_is_deterministic(run, dataset, tmp_path):
    train.main(["--data", str(dataset), "--out", str(tmp_path / "models"), "--seed", "42"])
    again = predict.main(["--data", str(dataset), "--model", str(tmp_path / "models" / train.MODEL_FILE),
                          "--splits", *SPLITS, "--out", str(tmp_path / "results")])
    pd.testing.assert_frame_equal(again.reset_index(drop=True), run["predictions"].reset_index(drop=True))


def test_pipeline_runs_without_ground_truth_files(run, dataset, tmp_path):
    data = tmp_path / "data"
    for split in ("train", "test_id"):
        (data / split).mkdir(parents=True)
        shutil.copy(dataset / split / "events.csv", data / split / "events.csv")
    shutil.copy(dataset / "hosts.csv", data / "hosts.csv")
    train.main(["--data", str(data), "--out", str(tmp_path / "models")])
    preds = predict.main(["--data", str(data), "--model", str(tmp_path / "models" / train.MODEL_FILE),
                          "--splits", "test_id", "--out", str(tmp_path / "results")])
    expected = run["predictions"][run["predictions"].split == "test_id"].reset_index(drop=True)
    pd.testing.assert_frame_equal(preds.reset_index(drop=True), expected)


def test_scripts_do_not_reference_ground_truth_or_evaluation():
    for name in ("train.py", "predict.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for banned in ("ground_truth", "import evaluation", "from evaluation", "trap_type", "expected_stages"):
            assert banned not in text, (name, banned)


def test_evaluation_cli_accepts_the_predictions(run, dataset):
    results = evaluation_run.main(["--predictions", str(run["out"] / "results" / predict.PREDICTIONS_FILE),
                                   "--data", str(dataset), "--splits", *SPLITS,
                                   "--out", str(run["out"] / "evaluation")])
    detectors = results["detectors"]
    assert set(detectors) == {*DETECTORS, "0_majority"}
    for split in SPLITS:
        sizes = {d: detectors[d]["splits"][split]["n_entity_days"] for d in detectors}
        assert len(set(sizes.values())) == 1, sizes
        assert all(detectors[d]["splits"][split]["missing_predictions"] == 0 for d in DETECTORS)
