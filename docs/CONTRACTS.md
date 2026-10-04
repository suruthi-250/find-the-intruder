# Interface contracts

These are the formats each component must read and write. Validators in [`intruder/contracts.py`](../intruder/contracts.py), [`intruder/events.py`](../intruder/events.py) and [`evaluation/ground_truth.py`](../evaluation/ground_truth.py) enforce them, and [`tests/`](../tests) checks the validators.

**Isolation rule:** ground truth is read only by the `evaluation/` package. The detection package `intruder/` never imports `evaluation/` and never references ground-truth files, and both rules are tested. The event and host loaders refuse to load any file that contains ground-truth columns (`label`, `trap_type`, `expected_stages`, `attack_start`, `attack_end`, `scenario_id`, `stage`).

---

## 1. Data layout (what the generator produces)

```text
data/generated/
  hosts.csv                                # one shared host inventory for all splits
  train/     events.csv  ground_truth.csv  # benign + benign_trap only (model fitting)
  val/       events.csv  ground_truth.csv  # threshold calibration
  test_id/   events.csv  ground_truth.csv  # same scenario profile as val, new seed
  test_ood/  events.csv  ground_truth.csv  # held-out scenario profile (generalisation)
```

`ground_truth.csv` lists only entity-days that need explicit evaluation (traps and attacks). Ordinary benign
activity is not listed and is treated as `benign` by the evaluation harness.

### 1.1 `events.csv`: detector input, observations only

| column | type | rule |
|---|---|---|
| `timestamp` | ISO-8601 UTC | required |
| `source_ip` | IP | required, except for `SUDO_EVENT` / `NEW_ACCOUNT`; internal `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`; external only `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24` |
| `destination_ip` | IP | optional, same address rules |
| `host` | string | optional; host that logged the event |
| `username` | string | optional |
| `event_type` | enum | `FAILED_LOGIN`, `SUCCESS_LOGIN`, `WEB_REQUEST`, `CONNECTION`, `SUDO_EVENT`, `NEW_ACCOUNT`, `EXTERNAL_TRANSFER`, `INTERNAL_TRANSFER` |
| `status` | enum | `SUCCESS`, `FAILURE`, `ALLOW`, `DENY` |
| `port` | int 0–65535 | optional |
| `bytes` | int ≥ 0 | optional (empty means 0) |
| `http_status` | int 100–599 | optional, `WEB_REQUEST` only |
| `request_marker` | enum | `NONE` (default), `SUSPECT_SQLI`, `SUSPECT_PATH_TRAVERSAL`, `SUSPECT_PROBE` |
| `privilege` | enum | `SUDO_EVENT` only: `ROUTINE` or `ELEVATED_SHELL`; must be empty otherwise |

Rows that violate the schema are skipped and counted by reason (`LoadReport.reasons`), and loading continues.

**Rules the data must follow (no leakage):**
- No column, value, IP range, host name, username or port may encode a stage or a label. Stages must only be inferable from patterns of ordinary observations.
- `EXTERNAL_TRANSFER` vs `INTERNAL_TRANSFER` only reflects where the destination is. Benign data must include external transfers too (for example updates or uploads), so the destination type alone can't separate the classes.
- `SUDO_EVENT` (including `ELEVATED_SHELL`) and `NEW_ACCOUNT` must also occur in benign activity.
- Benign activity and attacks must both appear at all times of day. The `off_hours` feature is context and must not be enough to separate the classes. `test_ood` must change the timing.
- Attacker and benign external addresses come from the same documentation ranges, so an IP address never reveals the label.
- `request_marker` values must also appear (rarely) in benign traffic, as false positives a web filter would raise.

### 1.2 `hosts.csv`: plain inventory

A single file at `data/generated/hosts.csv`, shared by all splits, with exactly the columns `host,ip`, for example `web-01,10.0.0.10`. It only maps host names to IP addresses. Any extra column is rejected. Load it with `load_hosts()` and pass the result as `Config(inventory=...)`.

### 1.3 `ground_truth.csv`: evaluation only

| column | rule |
|---|---|
| `split` | `train`, `val`, `test_id`, `test_ood` |
| `scenario_id` | non-empty |
| `entity_id` | synthetic IP (the acting source, or a host IP) |
| `day` | `YYYY-MM-DD`; the evaluation unit is **entity × day** |
| `label` | `benign`, `benign_trap`, `failed_attack`, `successful_intrusion` |
| `trap_type` | `benign_trap` rows only: `typo_login`, `internal_backup`, `failed_external_auth`, `noisy_benign` |
| `attack_start`, `attack_end` | attack rows only, ISO-8601; both on `day`; end ≥ start |
| `expected_stages` | attack rows only; pipe-separated, unique, canonical order |
| `profile` | `base` or `shifted` |

Additional rules: no duplicate `(split, entity_id, day)` rows, and `train` contains no attack rows. A `failed_attack` may only expect `RECON` / `CRED_ATTACK`. A `successful_intrusion` must expect `COMPROMISE`. Entity-days that are not listed count as `benign`.

### 1.4 Label definitions

| label | meaning |
|---|---|
| `benign` | ordinary authentication, web and internal traffic |
| `benign_trap` A, `typo_login` | a user mistypes their password a few times, then signs in normally |
| `benign_trap` B, `internal_backup` | a legitimate high-volume transfer to an internal backup destination |
| `benign_trap` C, `failed_external_auth` | low-volume, widely spaced failed logins from an external source (for example a misconfigured client); never succeeds |
| `benign_trap` D, `noisy_benign` | other legitimate high-volume or bursty activity that may look unusual |
| `failed_attack` | sustained hostile activity that never reaches `COMPROMISE` |
| `successful_intrusion` | hostile activity that reaches `COMPROMISE`, possibly followed by later stages |

The generator author decides the concrete behaviour of the attack scenarios. This document doesn't specify it.

### 1.5 Variation requirements (for the generalisation test)

Seeded and reproducible (`--seed`), but varied within and between splits: source IPs, event counts, rates, timing, ports, number of targeted users, and benign background volume. `test_ood` uses a scenario profile (`profile = shifted`) that `train`/`val` never contain.

---

## 2. Feature contract (`intruder/features.py`, implemented)

There is one row per `(entity, window_start)`. Windows are 5 minutes long and half-open (`[10:00, 10:05)`). Each row also carries `entity`, `day`, `window_start` and `first_eid`.

| feature | definition |
|---|---|
| `failed_logins` | count of `FAILED_LOGIN` |
| `successful_logins` | count of `SUCCESS_LOGIN` |
| `distinct_users` | distinct usernames across login events |
| `distinct_ports` | distinct ports across `CONNECTION` events |
| `denied_connections` | network events with status `DENY` |
| `web_requests` | count of `WEB_REQUEST` |
| `web_4xx` | web requests with `http_status` 400–499 |
| `suspicious_requests` | web requests with `request_marker != NONE` |
| `outbound_internal_bytes` | allowed network bytes to internal destinations |
| `outbound_external_bytes` | allowed network bytes to external destinations |
| `sudo_routine` | `SUDO_EVENT` with `ROUTINE` |
| `sudo_elevated` | `SUDO_EVENT` with `ELEVATED_SHELL` |
| `new_accounts` | count of `NEW_ACCOUNT` |
| `event_count` | events in the window |
| `event_rate` | `event_count / window_min` (events per minute) |
| `failure_ratio` | `(failed_logins + denied_connections + web_4xx) / event_count` |
| `off_hours` | 1 if the window starts outside business hours (context only) |

Model input: `model_matrix(features)`, which applies `log1p` to the count and byte columns and leaves `failure_ratio` and `off_hours` unchanged.

**Attribution:** a `SUDO_EVENT` without a source IP is credited to the latest `SUCCESS_LOGIN` of the same user on the same host within `session_hours`. A `NEW_ACCOUNT` is credited to the latest attributed sudo on that host within `account_link_min`. If neither exists, the event falls back to the host's IP in `hosts.csv`, or to `host:<name>`. Attribution is only bookkeeping and doesn't make an event suspicious.

---

## 3. Anomaly model interface (to be implemented)

- **Fit:** use windows from the `train` split only, which are all benign. No labels are used.
- **Score output:** `entity`, `window_start`, `anomaly_score`. Exactly one row per `(entity, window_start)`. `anomaly_score` is in **[0, 1]** and **higher = more anomalous**. Note that scikit-learn's `score_samples` runs the other way (higher = more normal), so it must be inverted and normalised. Check this with `check_score_direction(score_fn, typical, atypical)`.
- **Explanation output** (a separate layer, not produced by the model itself): `entity`, `window_start`, `feature`, `value`, `baseline_median`, `baseline_scale`, `deviation`. Each row describes how far one feature in a window sits from the **benign training baseline**. The anomaly model doesn't say which features caused a score, so the explanation must be worded as "deviates from baseline", not "caused the score".
- Validators: `validate_window_scores`, `validate_explanations`.

## 4. Stage signals interface (to be implemented)

The six stages, in canonical order:

| stage | meaning |
|---|---|
| `RECON` | discovery of services or resources |
| `CRED_ATTACK` | attempts to obtain valid credentials |
| `COMPROMISE` | the actor gains access with a valid account |
| `PRIV_ESC` | the actor obtains elevated privileges |
| `PERSISTENCE` | the actor arranges continued access |
| `EXFIL` | data leaves the environment in a way that departs from the baseline |

Stage signals are inferred from observations only, and are signals rather than ground truth. Output has one row per signal: `entity`, `ts`, `stage`, `eid` (the supporting event id in `events.csv` order), and `evidence` (non-empty text, for example a file and line reference). Validator: `validate_stage_signals`.

## 5. Progression interface (to be implemented)

**Input:** stage signals plus window scores for each entity, and the config (window length, maximum progression window, any stage weights).

**Output:** one row per `(entity, day)`:

| column | representation |
|---|---|
| `entity`, `day` | key |
| `stages_detected` | pipe-separated stages **in the order they were observed** (repeats allowed), e.g. `RECON\|CRED_ATTACK\|COMPROMISE` |
| `stage_times` | pipe-separated ISO-8601 timestamps aligned one-to-one with `stages_detected`, non-decreasing |
| `stages_completed` | number of distinct detected stages (0–6), shown as "3 / 6 stages" |
| `current_stage` | the most advanced detected stage in canonical order, or empty |
| `next_expected_stage` | the canonical stage after `current_stage`; empty if there is none or if `current_stage` is `EXFIL` |
| `progression_score` | 0–100 |
| `first_warning_ts` | time of the first suspicious signal for this entity-day, or empty |

The time gaps between stages can be derived from `stage_times`. A sequence observed in reverse canonical order is still a valid *representation*. How it is scored is up to the implementation. Validator: `validate_progression`.

## 6. Predictions interface (input to evaluation)

One row per `(split, detector, entity, day)` for **every** entity-day the detector processed, including Normal ones:

| column | rule |
|---|---|
| `split` | one of the four splits |
| `detector` | `A_anomaly_only`, `B_stage_count`, `C_progression` |
| `entity`, `day` | key |
| `risk_score` | 0–100 |
| `predicted_class` | `Normal`, `Suspicious`, `Failed attack`, `Intrusion` |
| `alert` | boolean; must equal `predicted_class != "Normal"` |
| `first_warning_ts` | required when `alert` is true |
| `current_stage` | a stage or empty (always empty for `A_anomaly_only`) |
| `stages_completed` | 0–6 |
| `explanation` | free text |

Validator: `validate_predictions`.

## 7. Evaluation (`evaluation/`, implemented)

```bash
py -m evaluation.run --predictions results/predictions.csv --data data/generated --out results
```

- **Detectors compared on identical entity-days:** `0_majority` (always Normal, added automatically), `A_anomaly_only`, `B_stage_count` (stages counted, ignoring order and timing), and `C_progression`.
- **Binary task:** attack = `failed_attack` + `successful_intrusion`; benign = `benign` + `benign_trap`. The report gives TP/FP/FN/TN, precision, recall, F1 = 2TP / (2TP + FP + FN), and FPR = FP / (FP + TN). A metric with a zero denominator is reported as `null`, not 0.
- **Per label:** the alert rate for each label.
- **Traps:** the count and FP rate for each `trap_type`.
- **Four-class:** a confusion matrix of label × predicted class, plus per-label accuracy, where the correct class is Normal for `benign` and `benign_trap`, Failed attack for `failed_attack`, and Intrusion for `successful_intrusion`. `Suspicious` counts as an alert but is never the correct four-class answer.
- **Detection latency** = `first_warning_ts − attack_start`, over detected attacks. It can be slightly negative when the first warning falls in a window that started before the labelled start.
- **Early-warning lead time** = `attack_end − first_warning_ts`, over detected attacks; `warned_before_end_rate` is the share with positive lead time.
- **Generalisation:** every metric is reported for `test_id` and `test_ood`, plus `generalisation_gap_f1 = F1(test_id) − F1(test_ood)`.
- **Counterfactual check** (part of the ablation): compare `progression_score` for an intact sequence and for the same sequence with a stage removed or reordered. The progression implementation supplies these scores through its own tests.
- Missing predictions for listed entity-days count as not alerted and are reported as `missing_predictions`.
