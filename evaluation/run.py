"""Score a predictions file against ground truth.

    py -m evaluation.run --predictions results/predictions.csv --data data/generated --out results

Reads <data>/<split>/ground_truth.csv for each requested split and writes metrics.json and results_table.csv/.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .ground_truth import load_ground_truth
from .metrics import evaluate, results_table


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="folder containing one sub-folder per split")
    ap.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    truth = load_ground_truth([args.data / split / "ground_truth.csv" for split in args.splits])
    predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    results = evaluate(predictions, truth, splits=tuple(args.splits))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = results_table(results)
    table.to_csv(args.out / "results_table.csv", index=False)
    (args.out / "results_table.md").write_text(table.to_markdown(index=False) if _has_tabulate() else
                                               table.to_string(index=False), encoding="utf-8")
    print(table.to_string(index=False))
    return results


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
