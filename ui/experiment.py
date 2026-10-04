"""EVALUATE pages: Evaluation and About the Model. Every number comes from results/metrics.json."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from intruder.contracts import DETECTORS, STAGES
from intruder.features import FEATURES
from intruder.prediction import ANOMALY_NORMAL_MAX, PROGRESSION_INTRUSION_MIN

from . import data, nav
from .components import badge, footer, header, html, note, section, table
from .theme import palette, style_figure

ID, OOD = "test_id", "test_ood"
SET_LABEL = {ID: "Test ID", OOD: "Test OOD"}
HELP_EVAL = """
**Does the method work?**

- **Precision**: of the entity-days flagged, the share that were real attacks.
- **Recall**: of the real attacks, the share that were flagged.
- **F1**: the balance of precision and recall.
- **False positive rate**: the share of benign entity-days wrongly flagged.
- **Trap false positives**: benign look-alikes (typo logins, backups, noisy activity) that were flagged.
- **Detection latency**: true attack start to first warning. **Early warning lead time**: first warning to true
  attack end.

**Test ID** behaves like the training data. **Test OOD** has deliberately shifted benign behaviour.
Ground truth is used only here, to score the detectors.
"""


def _b(m: dict, det: str, split: str) -> dict:
    return m["detectors"][det]["splits"][split]["binary"]


def _timing(m: dict, det: str, split: str) -> dict:
    return m["detectors"][det]["splits"][split]["timing"]["all_attacks"]


def _traps(m: dict, det: str, split: str) -> tuple[int, int]:
    t = m["detectors"][det]["splits"][split]["traps"].values()
    return sum(x["false_positives"] for x in t), sum(x["n"] for x in t)


def _overlap(split: str) -> float | None:
    p = data.predictions()
    p = p[p.split == split]
    b = p[p.detector == "B_stage_count"].set_index(data.KEYS).alert
    c = p[p.detector == "C_progression"].set_index(data.KEYS).alert
    either = (b | c).sum()
    return float((b & c).sum() / either) if either else None


def _interpretation(m: dict, split: str) -> tuple[str, list[str]]:
    a, b, c = (_b(m, d, split) for d in DETECTORS)
    f = lambda x: x["f1"] or 0  # noqa: E731
    ov = _overlap(split)
    if f(c) > f(b) + 0.005 and f(c) > f(a):
        short = "Yes. Attack Progression gives the best balance of precision and recall on this test set."
    elif f(c) > f(a):
        short = ("Partly. Adding attack evidence clearly beats behaviour anomaly alone; ordering the evidence adds "
                 "little beyond Attack Signals in the current configuration.")
    else:
        short = "Not on this test set: behaviour anomaly alone performs at least as well."
    pts = [f"<b>False positive rate</b> falls from <b>{data.fmt_pct(a['fpr'])}</b> (Behaviour Anomaly) to "
           f"<b>{data.fmt_pct(c['fpr'])}</b> (Attack Progression), and precision rises from {data.fmt_num(a['precision'])} "
           f"to {data.fmt_num(c['precision'])}.",
           f"<b>Recall</b> is {data.fmt_pct(a['recall'])} for Behaviour Anomaly and {data.fmt_pct(c['recall'])} for Attack "
           f"Progression: anomaly alone catches more attacks but raises {a['fp']:,} false alarms against {c['fp']:,}."]
    if ov is not None and ov > 0.95 and abs(f(c) - f(b)) < 0.005:
        pts.append(f"<b>Attack Signals and Attack Progression</b> flag almost the same entity-days (alert overlap "
                   f"{ov:.1%}). Ordering currently changes the verdict (for example Intrusion versus Failed attack) more "
                   f"than which entities are flagged.")
    else:
        pts.append(f"<b>Attack Progression versus Attack Signals:</b> F1 {data.fmt_num(c['f1'])} against "
                   f"{data.fmt_num(b['f1'])}.")
    pts.append("Thresholds are provisional (not yet calibrated on validation data); these results describe the current "
               "configuration.")
    return short, pts


def _metric_cells(m: dict, det: str, split: str) -> list[tuple[str, str]]:
    b, t = _b(m, det, split), _timing(m, det, split)
    fp, n = _traps(m, det, split)
    return [("Precision", data.fmt_num(b["precision"])), ("Recall", data.fmt_pct(b["recall"])),
            ("F1", data.fmt_num(b["f1"])), ("False positive rate", data.fmt_pct(b["fpr"])),
            ("Trap false positives", f"{fp} of {n}"),
            ("Detection latency", data.fmt_duration(t["latency"]["median_s"])),
            ("Early warning lead time", data.fmt_duration(t["lead_time"]["median_s"]))]


def evaluation() -> None:
    m = data.metrics()
    if ID not in m.get("splits", []) or OOD not in m.get("splits", []):
        note("Evaluation results for Test ID and Test OOD are not available yet. Run the evaluation step first.", "warn")
        return
    p = palette()
    header("Evaluate", "Does understanding attack progression improve detection?",
           "Compare the detector approaches under the same evaluation conditions.", HELP_EVAL)
    st.write("")
    note("<b>Research question.</b> Does adding temporal attack progression to behavioural anomaly detection improve "
         "intrusion detection and reduce false positives?")
    st.write("")
    split = st.segmented_control("Test set", [ID, OOD], default=ID, key="eval_split",
                                 format_func=lambda s: f"{SET_LABEL[s]} · {'same behaviour as training' if s == ID else 'shifted benign behaviour'}")
    split = split or ID
    html(f'<p class="small muted">{data.SPLIT_HELP[split]} All three detectors score exactly the same entity-days.</p>')

    cols = st.columns(3, gap="medium")
    for col, det in zip(cols, DETECTORS):
        main = det == data.DEFAULT_DETECTOR
        cells = "".join(f'<div style="display:flex;justify-content:space-between;padding:.32rem 0;border-bottom:1px solid '
                        f'var(--border)"><span class="small muted">{k}</span><b style="color:var(--text);font-variant-numeric:'
                        f'tabular-nums">{v}</b></div>' for k, v in _metric_cells(m, det, split))
        with col:
            html(f'<div class="panel{" accent" if main else ""}"><div class="eyebrow" style="color:'
                 f'{p["accent"] if main else p["muted"]}">{"Proposed" if main else "Baseline"} &middot; detector '
                 f'{data.SHORT[det]}</div><h4 style="margin-top:.4rem;font-size:1.15rem">{data.DETECTOR_NAME[det]}'
                 f'</h4><p class="small" style="min-height:2.8em">{data.DETECTOR_DESC[det]}</p>'
                 f'<div style="margin-top:.6rem">{cells}</div></div>')

    short, pts = _interpretation(m, split)
    section("What this shows")
    note(f'<b style="font-size:1.02rem">{short}</b><ul style="margin:.5rem 0 0 1.1rem;padding:0">'
         + "".join(f"<li>{x}</li>" for x in pts) + "</ul>")

    section("Side by side", f"{SET_LABEL[split]}. Higher is better except false positive rate.")
    with st.container():
        fig = go.Figure()
        keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("fpr", "False positive rate")]
        for det in DETECTORS:
            b = _b(m, det, split)
            fig.add_trace(go.Bar(name=data.DETECTOR_NAME[det], x=[l for _, l in keys], y=[b[k] or 0 for k, _ in keys],
                                 marker_color=p["detector"][det],
                                 text=[data.fmt_pct(b[k]) if k in ("recall", "fpr") else data.fmt_num(b[k]) for k, _ in keys],
                                 textposition="outside", textfont=dict(size=10)))
        fig.update_layout(barmode="group", bargap=.28, yaxis=dict(range=[0, 1.12], tickformat=".0%"))
        st.plotly_chart(style_figure(fig, 330), width="stretch", config={"displayModeBar": False})
    section("Test ID versus Test OOD", "How each detector copes when benign behaviour shifts away from what it was "
                                       "trained on.")
    with st.container():
        rows = []
        for det in DETECTORS:
            r = {"Detector": data.DETECTOR_NAME[det]}
            for s in (ID, OOD):
                fp, n = _traps(m, det, s)
                r[f"F1 · {SET_LABEL[s]}"] = data.fmt_num(_b(m, det, s)["f1"])
                r[f"FPR · {SET_LABEL[s]}"] = data.fmt_pct(_b(m, det, s)["fpr"])
                r[f"Trap FPs · {SET_LABEL[s]}"] = f"{fp} of {n}"
            rows.append(r)
        table(pd.DataFrame(rows), numeric=tuple(k for k in rows[0] if k != "Detector"), strong=("Detector",),
              highlight=pd.Series([d == data.DEFAULT_DETECTOR for d in DETECTORS]))

    section("Benign look-alikes", "Legitimate activity designed to resemble an attack. Flagging one is a false positive.")
    trap_names = {"typo_login": "Typo login", "internal_backup": "Internal backup",
                  "failed_external_auth": "Failed external authentication", "noisy_benign": "Noisy benign activity"}
    trows = []
    for key, name in trap_names.items():
        r = {"Look-alike": name}
        for det in DETECTORS:
            t = m["detectors"][det]["splits"][split]["traps"][key]
            r[data.DETECTOR_NAME[det]] = f'{t["false_positives"]} of {t["n"]} flagged'
        trows.append(r)
    table(pd.DataFrame(trows), strong=("Look-alike",))

    section("More detail")
    with st.expander("Verdict breakdown against ground truth"):
        det = st.radio("Detector", DETECTORS, index=2, horizontal=True, key="eval_conf_det",
                       format_func=data.DETECTOR_NAME.get)
        cm = pd.DataFrame(m["detectors"][det]["splits"][split]["four_class"]["confusion"]).T
        cm.index = [i.replace("_", " ").capitalize() for i in cm.index]
        table(cm.reset_index().rename(columns={"index": "Ground truth"}).astype(str),
              numeric=tuple(cm.columns), strong=("Ground truth",))
    with st.expander("Timing details"):
        rows = []
        for det in DETECTORS:
            for s in (ID, OOD):
                t = _timing(m, det, s)
                rows.append({"Detector": data.DETECTOR_NAME[det], "Test set": SET_LABEL[s],
                             "Median detection latency": data.fmt_duration(t["latency"]["median_s"]),
                             "Median early warning lead time": data.fmt_duration(t["lead_time"]["median_s"]),
                             "Warned before attack ended": data.fmt_pct(t["warned_before_end_rate"])})
        table(pd.DataFrame(rows))
        html('<p class="small muted" style="margin-top:.4rem">Latency is first warning minus the true attack start; '
             'lead time is the true attack end minus the first warning. Computed over detected attacks.</p>')
    with st.expander("Full results table"):
        res = pd.read_csv(data.RESULTS / "results_table.csv")
        res["detector"] = res.detector.map(lambda d: data.DETECTOR_NAME.get(d, "Always normal (baseline)"))
        table(res.fillna("—").astype(str), nowrap=True)
    footer()


# ------------------------------------------------------------------------------------------------ About the Model
PIPELINE = [("Synthetic events", False), ("Event parsing", False), ("Entity attribution", False),
            ("5-minute behavioural windows", False), ("17 behavioural features", False), ("Isolation Forest", True),
            ("Attack-stage evidence", False), ("Temporal progression", True), ("Risk + verdict", False), ("Evaluation", False)]


def _pipe(steps, dashed_last: bool = False, per_row: int = 5) -> str:
    out = []
    for i, (t, hl) in enumerate(steps, 1):
        cls = "p" + (" hl" if hl else "") + (" dashed" if dashed_last and i == len(steps) else "")
        out.append(f'<div class="{cls}"><div class="n">{i:02d}</div><div class="t">{t}</div></div>')
    return f'<div class="pipe" style="--per-row:{per_row}">' + "".join(out) + "</div>"


def about() -> None:
    s = data.training_summary()
    header("Evaluate", "About the Model", "How does it work?")
    st.write("")
    html(_pipe(PIPELINE))

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(f'''<div class="panel"><div class="eyebrow">Behaviour Anomaly</div><h4 style="margin-top:.4rem">Isolation Forest</h4>
          <p>Learns what benign behaviour looks like and identifies behaviour that is unusual relative to that baseline.</p>
          <p class="small muted" style="margin-top:.6rem">Trained only on {s.get("windows", 0):,} benign five-minute windows
          from the training split &middot; {len(FEATURES)} behavioural features &middot; {s.get("n_estimators", "—")} trees
          &middot; random seed {s.get("random_state", "—")}. The anomaly score runs from 0 (ordinary) to 1 (as unusual as
          anything seen in training). It is not an attack probability.</p></div>''')
    with c2:
        html(f'''<div class="panel accent"><div class="eyebrow">Attack Progression</div><h4 style="margin-top:.4rem">Temporal progression</h4>
          <p>Checks whether observed attack-stage evidence forms a plausible sequence over time.</p>
          <p class="small muted" style="margin-top:.6rem">For each entity and day, stages are kept while they move forward
          through the attack chain; counting stops at the first step that goes backwards. Repeats count once and missing
          stages are tolerated. A progression score of {PROGRESSION_INTRUSION_MIN:.0f} or more (three of six stages in
          order) leads to an Intrusion verdict.</p></div>''')

    section("Attack-stage evidence", "Six stages of a typical intrusion, inferred from observed events by deterministic rules.")
    p = palette()
    cols = st.columns(3, gap="small")
    for i, stage in enumerate(STAGES):
        with cols[i % 3]:
            html(f'<div class="panel" style="margin-bottom:.7rem;border-top:3px solid {p["stage"][stage]}">'
                 f'<div class="eyebrow" style="color:{p["stage"][stage]}">Stage {i + 1}</div>'
                 f'<h4 style="margin-top:.3rem">{data.STAGE_NAME[stage]}</h4><p class="small">{data.STAGE_TEXT[stage]}</p></div>')
    html('<p class="small muted">Stage signals are evidence, not proof. Benign activity can resemble a single stage, '
         'which is why the order of stages matters.</p>')

    section("From evidence to a verdict")
    rows = pd.DataFrame({
        "Detector": [f"{data.DETECTOR_NAME[d]}" for d in DETECTORS],
        "Uses": ["Anomaly score", "Anomaly score + distinct attack stages (order ignored)",
                 "Anomaly score + temporal progression"],
        "Raises an alert when": [f"anomaly score reaches {ANOMALY_NORMAL_MAX:g}",
                                 "any attack-stage evidence, or high combined risk",
                                 "any attack-stage evidence, or high combined risk"]})
    table(rows, strong=("Detector",), highlight=pd.Series([d == data.DEFAULT_DETECTOR for d in DETECTORS]))
    html(f'<p class="small muted" style="margin-top:.5rem">Verdicts: {badge("Suspicious")} isolated evidence or unusual '
         f'behaviour &nbsp; {badge("Failed attack")} two or more stages, or reconnaissance plus a credential attack, that do '
         f'not qualify as an intrusion &nbsp; {badge("Intrusion")} later-stage evidence with enough risk, or for Attack '
         f'Progression three or more stages in order. Risk ranks entities from 0 to 100; '
         f'it is not a probability.</p>')

    section("How this could fit into a security environment" + '<span class="tagc">Conceptual architecture</span>',
            "The detection pipeline stays the same; only the data source and the delivery of results change.")
    html(_pipe([("Real security logs", False), ("Log parser / normaliser", False), ("Behavioural feature extraction", False),
                ("Trained anomaly model", True), ("Attack-stage inference", False), ("Temporal progression", True),
                ("Detection / investigation", False), ("Security team", False)], dashed_last=True, per_row=8))
    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html('<div class="panel"><h4>Current prototype</h4><ul style="margin:.4rem 0 0 1.1rem;padding:0;color:var(--text2)">'
             '<li>Synthetic data</li><li>Offline processing</li><li>Streamlit visualisation</li>'
             '<li>No automated response</li></ul></div>')
    with c2:
        html('<div class="panel"><h4>Potential production integration</h4><ul style="margin:.4rem 0 0 1.1rem;padding:0;'
             'color:var(--text2)"><li>Endpoint, network and authentication logs</li><li>A log pipeline feeding the same '
             'event schema</li><li>Scheduled batch feature computation</li><li>Model inference</li>'
             '<li>Analyst-facing alerting and investigation</li></ul>'
             '<p class="small muted" style="margin-top:.5rem">Not implemented in this prototype.</p></div>')

    with st.expander(f"The {len(FEATURES)} behavioural features"):
        rows = [{"Group": g, "Feature": data.FEATURE_LABEL[f], "Technical name": f}
                for g, feats in data.FEATURE_GROUPS.items() for f in feats]
        table(pd.DataFrame(rows), strong=("Feature",))
    st.write("")
    a, _ = st.columns([1.3, 4])
    with a:
        nav.button("See the evaluation", "evaluation", key="about_eval", primary=True)
    footer()
