# Find the Intruder: progression-aware behavioural anomaly detection

ALGOTHON'26 · Problem statement **ALG-CYBER-01, "Find the Intruder"**

> **Status: work in progress.** This README separates what is implemented and tested from what is planned. No detection results are reported yet, because none have been measured.

## Research question

> Does adding temporal attack progression to behavioural anomaly detection improve intrusion detection while reducing false positives from benign look-alike activity?

The planned contribution has four parts:
1. **Behavioural anomaly detection** flags unusual activity in 5-minute behavioural windows.
2. **Temporal attack progression** separates an attack that develops through stages from an isolated anomaly.
3. **Trap-aware evaluation** tests whether benign look-alikes (password typos, backups, noisy clients) cause false positives.
4. **Early warning and ablation** measure how early an attack is flagged, and whether temporal ordering helps compared with simply counting stages.

## Architecture

```mermaid
flowchart TD
    G[Synthetic scenario generator<br/>events.csv + hosts.csv]:::planned --> L[Event loader & validator]:::done
    G -. ground_truth.csv .-> E
    L --> F[5-minute behavioural features]:::done
    F --> M[Anomaly model<br/>Isolation Forest, benign-trained]:::planned
    F --> S[Stage signals<br/>RECON … EXFIL]:::planned
    M --> X[Explanation layer<br/>deviation from benign baseline]:::planned
    M --> R[Final risk & classification<br/>Normal / Suspicious / Failed attack / Intrusion]:::planned
    S --> P[Temporal progression<br/>order + time window + partial chains]:::planned
    P --> R
    R --> E[Evaluation harness<br/>A vs B vs C, traps, OOD, latency, lead time]:::done
    E --> D[Streamlit demo]:::planned
    classDef done fill:#d4edda,stroke:#2e7d32;
    classDef planned fill:#fff3cd,stroke:#b8860b;
```

Green = implemented and tested · yellow = planned. Ground truth (dashed line) only goes into the evaluation harness.

## Status

| Component | Status | Where |
|---|---|---|
| Event schema, loader and validation (malformed rows skipped and counted, synthetic-IP check, label-leak guard) | ✅ implemented, tested | `intruder/events.py` |
| Raw log parser (syslog/auth, web access, CSV/JSONL) | ✅ implemented | `intruder/parsers.py` |
| 5-minute behavioural features + session attribution | ✅ implemented, tested | `intruder/features.py` |
| Interface contracts and validators (scores, explanations, stages, progression, predictions) | ✅ implemented, tested | `intruder/contracts.py`, `docs/CONTRACTS.md` |
| Ground-truth validation | ✅ implemented, tested | `evaluation/ground_truth.py` |
| Evaluation harness (binary, per-trap, four-class, latency, lead time, OOD gap, majority baseline) | ✅ implemented, tested on hand-made tables | `evaluation/metrics.py`, `evaluation/run.py` |
| Synthetic scenario generator | ⏳ planned | `generator/` |
| Isolation Forest + explanation layer | ⏳ planned | |
| Stage signals | ⏳ planned | |
| Temporal progression, early warning, partial progression | ⏳ planned | |
| Risk classification and threshold calibration on `val` | ⏳ planned | |
| Ablation runs (A / B / C) and counterfactual | ⏳ planned; harness ready | |
| Streamlit demo | ⏳ planned | |

## Methodology

**Data.** The generator writes abstract, non-operational security events (`events.csv`), a host inventory (`hosts.csv`) and a separate `ground_truth.csv` for four splits: `train` (benign only), `val`, `test_id`, and `test_ood` (a held-out scenario profile). All addresses are RFC 1918 or RFC 5737 documentation ranges. The full schema is in [docs/CONTRACTS.md](docs/CONTRACTS.md).

**Features.** There are 17 per-entity features per 5-minute window: login successes and failures, distinct users and ports, denied connections, web 4xx and marked requests, internal and external outbound bytes, routine and elevated sudo, new accounts, event count and rate, failure ratio, and an off-hours flag. Destination class and off-hours are context features only.

**Anomaly model (planned).** An Isolation Forest fitted on benign `train` windows only. The score is normalised to [0, 1] with higher = more anomalous. A separate explanation layer reports how each feature deviates from the benign baseline. The model itself does not attribute its score to features.

**Progression (planned).** Stage signals are inferred from observations and combined into an order-sensitive, time-bounded progression per entity-day. It reports partial progress (for example "3 / 6 stages, current COMPROMISE, next PRIV_ESC") and the first warning time.

**Evaluation (harness implemented).** The unit is entity × day, and attack = `failed_attack` + `successful_intrusion`. Detectors compared on the same entity-days:

| ID | Detector |
|---|---|
| 0 | Majority baseline (always Normal) |
| A | Anomaly only |
| B | Stage counting without temporal order |
| C | Full progression-aware detector |

Reported metrics: precision, recall, F1, FPR, false positives for each trap type, four-class confusion, detection latency (`first_warning − attack_start`), early-warning lead time (`attack_end − first_warning`), and the in-distribution vs OOD F1 gap. Thresholds are to be calibrated on `val` only and then frozen for the test splits.

## Reproduce

Requires Python 3.11+ and `pandas`, `numpy`, `scikit-learn`, `pytest` (`streamlit` and `plotly` for the demo).

```bash
py -m pip install pandas numpy scikit-learn pytest streamlit plotly
```

```bash
py -m pytest -q
```

Score a predictions file once the detectors exist:

```bash
py -m evaluation.run --predictions results/predictions.csv --data data/generated --out results
```

The generator and training commands will be added here once those components exist.

## Results

**Not measured yet.** This section will hold the A / B / C results table (from `results/results_table.md`), the trap false-positive table, and latency and lead-time figures.

## Limitations (known so far)

- Data is synthetic. Results will show behaviour on the generated scenario families and may not carry over to real networks.
- The evaluation unit is entity × day, so attacks are assumed not to cross midnight.
- Session attribution is heuristic: it uses the latest login of the same user on the same host.
- `request_marker` is an abstract stand-in for a web-filter verdict. No payload inspection is performed.

## AI / tool disclosure

- AI assistance: Claude (Anthropic) via Claude Code was used to draft the event loader, feature extraction, interface contracts, evaluation harness, tests and this README. All of it was reviewed by the author.
- Libraries: pandas, NumPy, scikit-learn, pytest; Streamlit and Plotly are planned for the demo.
- Datasets: none external. All data is synthetic.
