"""OVERVIEW and INVESTIGATE pages: Dashboard, Intruders, Attack Stories, Evidence."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from intruder.contracts import DETECTORS, STAGES
from intruder.prediction import ANOMALY_NORMAL_MAX

from . import data, nav
from .components import (badge, band, chips, entity_picker, entity_table, footer, header, html, kpis, note,
                         progress6, section, table, timeline)
from .theme import palette, style_figure

HELP_DASHBOARD = """
**What is happening in this dataset?**

- **Security overview** counts entity-days (one address on one day) by verdict from **Attack Progression**, the
  detector this project proposes.
- **Unusual is not malicious** shows how many entity-days answer *yes* to each of the detector's questions.
- **Attack activity** shows which attack stages were actually observed.
- **High-risk entities** is where to start an investigation.

Switch dataset or theme at the bottom of the sidebar.
"""
HELP_INTRUDERS = """
**Who needs attention?**

- **Risk** (0–100) ranks entities; it is not a probability.
- **Verdict** is the detector's conclusion, written out in words.
- **Attack activity** lists the attack stages observed for the entity.
- **Progression** is how far those stages form an ordered attack (score and stages in order).
- **Detection latency** is the time from the entity's first activity that day to the first warning.

Use **Investigate** to open an entity's attack story.
"""
HELP_STORY = """
**What happened, in order.**

- The strip shows the six stages of a typical intrusion. A tick means the stage was observed; *Out of order* means it
  appeared but did not fit the attack sequence; a dash means it was not observed.
- For a **failed attack** the strip marks where the attack stopped.
- The **detection timeline** lists each step at the time it was observed. Open a step to see its evidence and the
  five-minute processing window behind it.
"""
HELP_EVIDENCE = """
**Why was this flagged?**

1. **Detection summary**: the four questions the detector asks, answered for this entity.
2. **Behavioural profile**: what the entity did during its most unusual five minutes, next to typical benign behaviour.
3. **Attack-stage evidence**: each stage and the events behind it.
4. **Raw events**: collapsed; open them only if you want to verify every detail.
"""


def _row_actions(row, key: str) -> None:
    a, b, _ = st.columns([1.3, 1.3, 3])
    with a:
        if st.button("Open attack story", key=f"{key}_story", type="primary", width="stretch"):
            data.set_focus(row.entity, row.day)
            nav.go("stories")
    with b:
        if st.button("Open evidence", key=f"{key}_evidence", width="stretch"):
            data.set_focus(row.entity, row.day)
            nav.go("evidence")


# ------------------------------------------------------------------------------------------------ Dashboard
def _hero(split: str) -> None:
    p = palette()
    ws, sig, _ = data.outputs(split)
    q = data.question_counts(split)
    unusual_windows = int((ws.anomaly_score >= ANOMALY_NORMAL_MAX).sum())
    layers = [("01", "Behaviour", "five-minute activity windows", f"{len(ws):,}", "windows", p["muted"]),
              ("02", "Anomaly", "unusual compared with benign training", f"{unusual_windows:,}", "windows", p["warn"]),
              ("03", "Attack signals", "stage evidence found in the events", f"{len(sig):,}", "signals", p["stage"]["CRED_ATTACK"]),
              ("04", "Progression", "two or more stages in attack order", f"{q['ordered']:,}", "entity-days", p["accent"]),
              ("05", "Verdict", "intrusions after weighing the evidence", f"{q['intrusion']:,}", "intrusions", p["crit"])]
    stack = "".join(
        f'<div class="layer{" final" if i == 4 else ""}" style="--i:{i};--c:{c}"><div class="ix">{n}</div>'
        f'<div class="tx"><b>{t}</b><span>{d}</span></div><div class="vv">{v}<small>{u}</small></div></div>'
        for i, (n, t, d, v, u, c) in enumerate(layers))
    calm = "calm" if st.session_state.get("reduce_motion") else ""
    html(f'''<div class="hero {calm}"><div class="wrap">
      <div><div class="brand">Find the Intruder <span>&nbsp;&middot;&nbsp; Progression-aware ML intrusion detection</span></div>
        <div class="h1">See the anomaly.<br>Understand the progression.<br><em>Find the intruder.</em></div>
        <div class="lead">Find the Intruder combines behavioural anomaly detection with attack-stage evidence and
        temporal progression to distinguish real intrusions from benign look-alikes.</div>
        <div style="height:4.2rem"></div></div>
      <div class="stack" aria-label="Detection pipeline with counts for the selected dataset">{stack}
        <div class="small muted" style="margin-top:.2rem;text-align:right">Live counts &middot; {data.SPLITS[split]}</div></div>
    </div></div>''')
    with st.container(key="hero_cta"):
        a, b, _ = st.columns([1.15, 1.25, 3.6])
        with a:
            nav.button("Explore detections", "intruders", key="hero_explore", primary=True)
        with b:
            nav.button("Take the quick tour", "tour", key="hero_tour")


def dashboard() -> None:
    split = data.current_split()
    t = data.entity_days(split)
    p = palette()
    _hero(split)

    left, right = st.columns([4.1, 1.65], vertical_alignment="bottom")
    with left:
        section("Security overview", f"{data.SPLITS[split]} · {data.SPLIT_HELP[split]} Verdicts from Attack Progression.")
    with right, st.container(key="pagehelp"):
        with st.popover("What am I looking at?", icon=":material/help:", width="stretch"):
            st.markdown(HELP_DASHBOARD)
    vc = t.C_predicted_class.value_counts()
    kpis([("Intrusions", f"{int(vc.get('Intrusion', 0)):,}", "progressed past initial access", p["crit"]),
          ("Failed attacks", f"{int(vc.get('Failed attack', 0)):,}", "attack-like, not successful", p["high"]),
          ("Suspicious entities", f"{int(vc.get('Suspicious', 0)):,}", "unusual, no clear attack pattern", p["warn"]),
          ("Entities analysed", f"{len(t):,}", f"{t.entity.nunique():,} addresses over {t.day.nunique()} days", p["accent"])])

    q = data.question_counts(split)
    section("Unusual is not the same as malicious",
            "The detector asks four questions. Each count is the number of entity-days answering yes.")
    steps = [("01", "Is the behaviour unusual?", q["unusual"], f"anomaly score of {ANOMALY_NORMAL_MAX:g} or more"),
             ("02", "Is there evidence of attack activity?", q["signals"], "at least one attack-stage signal"),
             ("03", "Do the signals form a progression?", q["ordered"], "two or more stages in attack order"),
             ("04", "Final verdict: intrusion", q["intrusion"], "after weighing all the evidence")]
    html('<div class="qflow">' + "".join(
        f'<div class="qf{" last" if i == 3 else ""}"><div class="n">QUESTION {n}</div><div class="q">{qq}</div>'
        f'<div class="c">{c:,}</div><div class="d">{d}</div></div>' for i, (n, qq, c, d) in enumerate(steps)) + "</div>")

    section("Attack activity", "Attack-stage signals observed in this dataset, by stage.")
    act = data.stage_activity(split)
    c1, c2 = st.columns([1.55, 1], gap="large")
    with c1:
        fig = go.Figure(go.Bar(
            y=[data.STAGE_NAME[s] for s in act.stage], x=act.entities, orientation="h",
            marker_color=[p["stage"][s] for s in act.stage], text=act.entities, textposition="outside",
            hovertemplate="%{y}: %{x} entity-days<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="entity-days with the stage", bargap=.35)
        st.plotly_chart(style_figure(fig, 290), width="stretch", config={"displayModeBar": False})
    with c2:
        rows = pd.DataFrame({
            "Stage": [f'<span class="chip" style="--c:{p["stage"][s]}"><i></i>{data.STAGE_NAME[s]}</span>' for s in act.stage],
            "Signals": [f"{v:,}" for v in act.signals], "Entity-days": [f"{v:,}" for v in act.entities]})
        table(rows, numeric=("Signals", "Entity-days"))
        if (act.signals == 0).any():
            missing = ", ".join(data.STAGE_NAME[s] for s in act.stage[act.signals == 0])
            html(f'<p class="small muted" style="margin-top:.5rem">Not observed in this dataset: {missing}.</p>')

    section("High-risk entities", "Highest Attack Progression risk first. Investigate an entity to see its attack story.")
    entity_table(t[t.C_alert].reset_index(drop=True), data.DEFAULT_DETECTOR, key=f"dash_{split}", page_size=8)
    a, _ = st.columns([1.2, 4])
    with a:
        nav.button("See all intruders", "intruders", key="dash_all")
    footer()


# ------------------------------------------------------------------------------------------------ Intruders
def _sync_split() -> None:
    st.session_state["split"] = st.session_state.get("int_split") or data.current_split()


def intruders() -> None:
    split = data.current_split()
    t = data.entity_days(split)
    header("Investigate", "Intruders", "Entities showing unusual or attack-like behaviour.", HELP_INTRUDERS)
    st.write("")
    c1, c2, c3, c4 = st.columns([2.9, 1.3, 1.3, 1.3], vertical_alignment="bottom")
    verdicts = c1.multiselect("Verdict", data.VERDICTS, default=data.VERDICTS[:3], key="int_verdicts")
    min_risk = c2.slider("Minimum risk", 0, 100, 0, key="int_min_risk")
    query = c3.text_input("Entity", key="int_query", placeholder="Search address")
    st.session_state["int_split"] = split
    c4.selectbox("Dataset", list(data.SPLITS), format_func=data.SPLITS.get, key="int_split", on_change=_sync_split)
    with st.expander("Advanced: detector"):
        det = st.radio("Detector", DETECTORS, index=2, key="int_det", horizontal=True,
                       format_func=lambda d: data.DETECTOR_NAME[d] + (" (default)" if d == data.DEFAULT_DETECTOR else ""),
                       help="Attack Progression is the method under evaluation. The others are the research baselines.")
    s = data.SHORT[det]
    view = t[t[f"{s}_predicted_class"].isin(verdicts) & (t[f"{s}_risk_score"] >= min_risk)]
    if query:
        view = view[view.entity.str.contains(query.strip(), regex=False)]
    view = view.sort_values([f"{s}_risk_score", "anomaly"], ascending=False).reset_index(drop=True)
    html(f'<p class="small muted" style="margin:.6rem 0 .5rem 0">{len(view):,} entity-days &middot; '
         f'{data.DETECTOR_NAME[det]} &middot; {data.SPLITS[split]}</p>')
    entity_table(view, det, key=f"int_{split}_{s}_{'-'.join(verdicts)}_{min_risk}_{query}", page_size=15)
    footer()


# ------------------------------------------------------------------------------------------------ Attack stories
def _conclusion(row, eps: list[dict]) -> tuple[str, str]:
    v = row.C_predicted_class
    k = len(row.c_stages)
    if v == "Intrusion":
        start = data.fmt_time(eps[0]["start"]) if eps else "—"
        nxt = (f" The next stage of a typical attack, {data.STAGE_NAME[row.next_expected_stage].lower()}, was not "
               f"observed for this entity.") if row.next_expected_stage else ""
        return "crit", (f"<b>Intrusion.</b> Attack signals progressed past initial access: {k} of 6 stages occurred in "
                        f"attack order, starting at {start}.{nxt}")
    if v == "Failed attack":
        return "warn", f"<b>Failed attack.</b> {data.VERDICT_MEANING['Failed attack']}"
    if v == "Suspicious":
        why = "an isolated attack signal" if eps else "unusual behaviour alone"
        return "warn", (f"<b>Suspicious.</b> The alert comes from {why} (anomaly score {row.anomaly:.2f}); the signals "
                        f"do not form an attack sequence.")
    return "ok", "<b>Normal.</b> Nothing unusual enough to raise an alert."


def _plain_story(row, eps: list[dict]) -> str:
    if not eps:
        return (f"No attack-stage activity was observed for <b>{escape(row.entity)}</b> on this day. Its busiest five "
                f"minutes scored <b>{row.anomaly:.2f}</b> on the anomaly scale, where 1 matches the most unusual "
                f"behaviour seen in training.")
    names = list(dict.fromkeys(data.STAGE_NAME[e["stage"]].lower() for e in eps))
    listed = ", ".join(names[:-1]) + f" and then {names[-1]}" if len(names) > 1 else names[0]
    start, end = eps[0]["start"], eps[-1]["end"]
    text = (f"Between <b>{data.fmt_time(start)}</b> and <b>{data.fmt_time(end)}</b>, <b>{escape(row.entity)}</b> showed "
            f"{listed}. Attack Progression found <b>{len(row.c_stages)} of 6</b> stages in attack order (progression "
            f"score <b>{row.progression_score:.0f}</b>) and concluded <b>{row.C_predicted_class.lower()}</b> with risk "
            f"<b>{row.C_risk_score:.0f}</b>.")
    skipped = [s for s in dict.fromkeys(e["stage"] for e in eps) if s not in row.c_stages]
    if skipped:
        text += (f" {', '.join(data.STAGE_NAME[s] for s in skipped)} appeared out of order and was not counted as "
                 f"progression.")
    return text


def stories() -> None:
    split = data.current_split()
    t = data.entity_days(split)
    header("Investigate", "Attack Stories", "Understand how suspicious activity unfolded over time.", HELP_STORY)
    st.write("")
    row = entity_picker(t, key=f"story_pick_{split}")
    _, sig, _ = data.outputs(split)
    sig = sig[(sig.entity == row.entity) & (sig.day == row.day)]
    eps = data.episodes(sig)
    band(row)

    section("Attack progression", "The six stages of a typical intrusion, as observed for this entity.")
    progress6(sig, row.c_stages, row.C_predicted_class)
    tone, text = _conclusion(row, eps)
    note(text, tone)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section("Detection timeline", "Each step at the time it was observed. Open a step for its evidence.")
        if eps:
            timeline(split, row.entity, sig)
            html('<p class="small muted">Activity is internally summarised into five-minute windows so the model can '
                 'compare behaviour consistently. Each step lists the processing window it fell into.</p>')
        else:
            note("No attack-stage activity was observed for this entity on this day.")
    with right:
        section("In plain English")
        note(_plain_story(row, eps))
        section("How each detector judged it")
        rows = pd.DataFrame({
            "Detector": [f'<div><b>{data.DETECTOR_NAME[d]}</b><br><span class="small muted">research detector '
                         f'{data.SHORT[d]}</span></div>' for d in DETECTORS],
            "Verdict": [badge(row[f"{data.SHORT[d]}_predicted_class"]) for d in DETECTORS],
            "Risk": [f'{row[f"{data.SHORT[d]}_risk_score"]:.0f}' for d in DETECTORS]})
        table(rows, numeric=("Risk",), highlight=pd.Series([d == data.DEFAULT_DETECTOR for d in DETECTORS]))
        st.write("")
        with st.expander("Reveal ground truth (evaluation only)"):
            gt = data.ground_truth(split)
            hit = gt[(gt.entity_id == row.entity) & (gt.day == row.day)]
            if hit.empty:
                st.markdown("Not listed in the ground truth, so the evaluation treats this entity-day as **benign**. "
                            "The detectors never see ground truth.")
            else:
                rel = gt[gt.scenario_id.isin(hit.scenario_id)]
                table(pd.DataFrame({"Entity": rel.entity_id, "Label": rel.label.str.replace("_", " "),
                                    "Attack start": rel.attack_start.map(data.fmt_time),
                                    "Attack end": rel.attack_end.map(data.fmt_time)}))
                html('<p class="small muted" style="margin-top:.4rem">The detectors never see ground truth.</p>')
    st.write("")
    a, b, _ = st.columns([1.3, 1.3, 3])
    with a:
        nav.button("View the evidence", "evidence", key="story_evidence", primary=True)
    with b:
        nav.button("Back to intruders", "intruders", key="story_back")
    footer()


# ------------------------------------------------------------------------------------------------ Evidence
def evidence() -> None:
    split = data.current_split()
    t = data.entity_days(split)
    p = palette()
    header("Investigate", "Evidence", "Why did the detector flag this activity?", HELP_EVIDENCE)
    st.write("")
    row = entity_picker(t, key=f"evidence_pick_{split}")
    _, sig, _ = data.outputs(split)
    sig = sig[(sig.entity == row.entity) & (sig.day == row.day)].sort_values(["ts", "eid"])

    section("Detection summary", f"{escape(row.entity)} · {data.fmt_day(row.day)}")
    unusual = row.anomaly >= ANOMALY_NORMAL_MAX
    k = len(row.c_stages)
    progression = ("Consistent with an attack sequence" if k >= 2 else
                   "A single stage, not a sequence" if k == 1 else "No attack sequence")
    summary = pd.DataFrame({
        "Question": ["Behaviour", "Attack signals", "Progression", "Final verdict"],
        "Answer": [f'<b>{"Unusual" if unusual else "Within the normal range"}</b>',
                   chips(row.b_stages),
                   f"<b>{progression}</b>",
                   badge(row.C_predicted_class)],
        "Detail": [f"anomaly score {row.anomaly:.2f} (alert level {ANOMALY_NORMAL_MAX:g}; 1 = most unusual seen in training)",
                   f"{row.signals} signal{'s' if row.signals != 1 else ''} across {len(row.b_stages)} stage"
                   f"{'s' if len(row.b_stages) != 1 else ''}",
                   f"{k} of 6 stages in attack order · progression score {row.progression_score:.0f}",
                   f"Attack Progression · risk {row.C_risk_score:.0f} of 100"]})
    table(summary, strong=("Question",))

    section("Behavioural profile", "Behaviour observed during this entity's most unusual five-minute window, next to "
                                   "typical benign behaviour learned in training.")
    dev, top = data.deviations(split, row.entity, row.day)
    if dev.empty or top is None:
        note("No behaviour profile is available for this entity-day.")
    else:
        w0 = pd.Timestamp(top.window_start)
        note(f"Activity is summarised into five-minute windows so the model can compare behaviour consistently. This "
             f"is the window <b>{w0:%H:%M}&ndash;{w0 + pd.Timedelta(minutes=5):%H:%M} UTC</b>, anomaly score "
             f"<b>{top.anomaly_score:.2f}</b>. These are observed behavioural signals; the anomaly model does not say "
             f"which signal drove its score.")
        st.write("")
        dv = dev.set_index("feature")
        cols = st.columns(2, gap="medium")
        for i, (group, feats) in enumerate(data.FEATURE_GROUPS.items()):
            with cols[i % 2]:
                rows = pd.DataFrame({
                    "Signal": [data.FEATURE_LABEL[f] for f in feats],
                    "Observed": [data.feature_value(f, dv.at[f, "value"]) for f in feats],
                    "Typical benign": [data.feature_value(f, dv.at[f, "typical"]) for f in feats],
                    "": ['<span class="badge high">above typical</span>'
                         if dv.at[f, "value"] > dv.at[f, "typical"] and dv.at[f, "deviation"] > 1 else "" for f in feats]})
                html(f'<div class="grp" style="margin-top:.4rem">{group}</div>')
                table(rows, numeric=("Observed", "Typical benign"), strong=("Observed",))

    section("Attack-stage evidence", "Each stage of a typical intrusion and the events behind it.")
    stage_rows = []
    for stage in STAGES:
        s = sig[sig.stage == stage]
        if len(s):
            status = ('<span class="badge info"><i></i>In sequence</span>' if stage in row.c_stages
                      else '<span class="badge high"><i></i>Out of order</span>')
            stage_rows.append({"Stage": f'<span class="chip" style="--c:{p["stage"][stage]}"><i></i>'
                                        f'{data.STAGE_NAME[stage]}</span>',
                               "First seen": data.fmt_time(s.ts.min()), "Signals": f"{len(s)}",
                               "Latest evidence": str(s.evidence.iloc[-1]), "Status": status})
        else:
            stage_rows.append({"Stage": f'<span class="chip" style="--c:{p["faint"]}"><i></i>{data.STAGE_NAME[stage]}</span>',
                               "First seen": "—", "Signals": "0", "Latest evidence": "—",
                               "Status": '<span class="badge neutral"><i></i>Not observed</span>'})
    table(pd.DataFrame(stage_rows), numeric=("Signals",))

    section("Raw events", "Collapsed by default. Open to verify every detail.")
    ev = data.attributed_events(split)
    ev = ev[(ev.entity == row.entity) & (ev.day == row.day)].sort_values(["ts", "eid"])
    used = set(sig.eid)
    cols_present = [c for c in ("event_type", "status", "source_ip", "destination_ip", "host", "username", "port",
                                "bytes", "privilege") if c in ev.columns]
    with st.expander(f"Show raw events ({len(ev):,})"):
        shown = ev.head(300)
        raw = pd.DataFrame({"Time": shown.ts.map(data.fmt_time), "Event": shown.event_type,
                            **{c.replace("_", " ").capitalize(): shown[c].map(lambda v: "—" if v is None or pd.isna(v) else
                                                                               (f"{v:,.0f}" if isinstance(v, float) else str(v)))
                               for c in cols_present if c != "event_type"},
                            "Evidence": shown.eid.map(lambda e: '<span class="badge info"><i></i>stage signal</span>'
                                                      if e in used else "")})
        table(raw, nowrap=True)
        if len(ev) > 300:
            html(f'<p class="small muted">Showing the first 300 of {len(ev):,} events.</p>')
    st.write("")
    a, b, _ = st.columns([1.3, 1.3, 3])
    with a:
        nav.button("Back to the attack story", "stories", key="evidence_story")
    with b:
        nav.button("Back to intruders", "intruders", key="evidence_back")
    footer()
