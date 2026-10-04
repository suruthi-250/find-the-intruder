"""LEARN pages: User Guide and Quick Tour (plus `home`, kept as an alias of the overview dashboard)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from intruder.contracts import DETECTORS
from intruder.prediction import ANOMALY_NORMAL_MAX

from . import data, investigate, nav
from .components import badge, chips, footer, header, html, kpis, note, progress6, risk, table
from .theme import palette


def home() -> None:
    investigate.dashboard()


# ------------------------------------------------------------------------------------------------ User Guide
GUIDE = [
    ("Getting started", [
        ("What is Find the Intruder?",
         "A research prototype for ML-based intrusion detection. It learns what normal behaviour looks like, notices "
         "when an entity (a network address) behaves unusually, and checks whether that behaviour lines up into an "
         "attack. It explains every conclusion with evidence. It is not a SIEM or log-management tool.", ""),
        ("How does detection work?",
         "Three questions, in order: is the behaviour unusual? Is there evidence of attack activity? Do those signals "
         "form a plausible progression over time? The answers are combined into a risk score and a verdict.",
         "Isolation Forest on 17 behavioural features per five-minute window; six rule-based attack-stage detectors; "
         "an order-sensitive progression score; thresholds are provisional until calibrated on validation data."),
    ]),
    ("Key ideas", [
        ("What is a behavioural anomaly?",
         "Behaviour that differs from what the model learned as normal, such as many failed logins or contact with many "
         "network ports in a short time. Unusual does not automatically mean malicious: backups and typos are unusual too.",
         "The anomaly score runs from 0 (ordinary) to 1 (as unusual as anything seen in training). It is not a "
         "probability of attack."),
        ("What is attack-stage evidence?",
         "Signs in the events that match one of six stages of a typical intrusion: reconnaissance, credential attack, "
         "compromise, privilege escalation, persistence and exfiltration.",
         "Stages are inferred by deterministic rules from observed events. They are evidence, not proof."),
        ("What is attack progression?",
         "Whether the attack stages happen in the order a real attack would follow. A burst of failed logins on its own "
         "is different from probing, then guessing passwords, then logging in, then gaining privileges.",
         "Stages are kept while they move forward through the chain for one entity on one day; repeats count once and "
         "missing stages are allowed."),
        ("What are the five-minute windows?",
         "Behind the scenes, activity is summarised into five-minute windows so the model can compare behaviour "
         "consistently. You never need to think in windows: the app shows the time something happened, and lists the "
         "window only as a detail.", ""),
    ]),
    ("Verdicts", [
        ("What does Suspicious mean?", data.VERDICT_MEANING["Suspicious"], ""),
        ("What does Failed Attack mean?", data.VERDICT_MEANING["Failed attack"], ""),
        ("What does Intrusion mean?", data.VERDICT_MEANING["Intrusion"], ""),
    ]),
    ("Investigating", [
        ("How do I investigate an entity?",
         "Open Intruders (or the high-risk table on the Dashboard) and choose Investigate on any row. That opens its "
         "Attack Story; from there, View the evidence explains the verdict. The sidebar shows which entity you are "
         "investigating.", ""),
        ("How do I read an Attack Story?",
         "The strip at the top shows the six stages: a tick means observed, a dash means not observed, and \"Attack "
         "stopped\" marks where a failed attack ended. The timeline below lists each step at the time it happened; open "
         "a step for its evidence.", ""),
        ("How do I read Evidence?",
         "Start with the detection summary (four questions, four answers). Then the behavioural profile compares what "
         "the entity did with typical benign behaviour. Attack-stage evidence and raw events come last.", ""),
    ]),
    ("The research", [
        ("What are Behaviour Anomaly, Attack Signals and Attack Progression?",
         "Three detectors that score exactly the same entities. Behaviour Anomaly uses only the anomaly score. Attack "
         "Signals adds how many distinct attack stages appeared. Attack Progression adds whether those stages occur in "
         "attack order; it is the method under evaluation and the default everywhere in the app.",
         "In the research write-up they are detectors A, B and C."),
        ("What does ID mean?", "In-distribution: test data with the same general behavioural profile as training.", ""),
        ("What does OOD mean?",
         "Out-of-distribution: test data with deliberately shifted benign behaviour. It checks whether the detectors "
         "still work when everyday behaviour changes.",
         "The shifted profile changes benign event rates, failure rates, transfer sizes and the shape of the benign "
         "look-alikes. Attack behaviour is unchanged."),
    ]),
]


def guide() -> None:
    header("Learn", "User Guide", "How do I use it? Plain-English answers, with technical detail folded away.")
    links, n = [], 0
    for _, items in GUIDE:
        for q, _, _ in items:
            n += 1
            links.append(f'<a href="#q{n}">{q}</a>')
    html(f'<div class="toc">{"".join(links)}</div>')
    n = 0
    for group, items in GUIDE:
        cards = ""
        for q, a, tech in items:
            n += 1
            more = f"<details><summary>Technical detail</summary><p>{tech}</p></details>" if tech else ""
            cards += f'<div class="panel" id="q{n}"><div class="q">{q}</div><div class="a">{a}</div>{more}</div>'
        html(f'<div class="grp">{group}</div><div class="qa">{cards}</div>')
    st.write("")
    a, b, _ = st.columns([1.3, 1.3, 3])
    with a:
        nav.button("Start investigating", "intruders", key="guide_start", primary=True)
    with b:
        nav.button("Take the quick tour", "tour", key="guide_tour")
    footer()


# ------------------------------------------------------------------------------------------------ Quick Tour
TOUR = [
    ("Dashboard", "What is happening?", "dashboard",
     "The Dashboard summarises the dataset: how many intrusions, failed attacks and suspicious entities Attack "
     "Progression found, and how the detector narrows unusual behaviour down to real intrusions."),
    ("Intruders", "Who needs attention?", "intruders",
     "Intruders ranks every entity by risk, with its verdict, attack activity and progression. Choose Investigate on "
     "any row to open it."),
    ("Attack Story", "What happened?", "stories",
     "The Attack Story shows which of the six attack stages occurred and in what order, then a timeline of each step "
     "at the time it happened."),
    ("Evidence", "Why was it detected?", "evidence",
     "Evidence answers the detector's questions for one entity, compares its behaviour with typical benign behaviour, "
     "and keeps raw events one click away."),
    ("Evaluation", "Does the method work?", "evaluation",
     "Evaluation compares Behaviour Anomaly, Attack Signals and Attack Progression on the same test data, both "
     "in-distribution and with shifted benign behaviour."),
    ("You're ready", "Start with the highest-risk entities.", "intruders",
     "That is the whole flow: overview, who needs attention, what happened, why, and whether the method works."),
]


def _preview(i: int) -> None:
    split = data.current_split()
    t = data.entity_days(split)
    row = t.iloc[0]
    p = palette()
    if i == 0:
        vc = t.C_predicted_class.value_counts()
        kpis([("Intrusions", f"{int(vc.get('Intrusion', 0)):,}", "", p["crit"]),
              ("Failed attacks", f"{int(vc.get('Failed attack', 0)):,}", "", p["high"]),
              ("Suspicious entities", f"{int(vc.get('Suspicious', 0)):,}", "", p["warn"]),
              ("Entities analysed", f"{len(t):,}", data.SPLITS[split], p["accent"])])
    elif i == 1:
        top = t.head(3)
        table(pd.DataFrame({"Entity": top.entity, "Risk": [risk(r.C_risk_score, r.C_predicted_class) for r in top.itertuples()],
                            "Verdict": top.C_predicted_class.map(badge), "Attack activity": top.c_stages.map(chips)}))
    elif i in (2, 3):
        data.set_focus(row.entity, row.day)
        _, sig, _ = data.outputs(split)
        sig = sig[(sig.entity == row.entity) & (sig.day == row.day)]
        if i == 2:
            html(f'<p class="small muted">Example: <b style="color:var(--text)">{row.entity}</b> &middot; '
                 f'{data.fmt_day(row.day)}</p>')
            progress6(sig, row.c_stages, row.C_predicted_class)
        else:
            unusual = "Unusual" if row.anomaly >= ANOMALY_NORMAL_MAX else "Within the normal range"
            table(pd.DataFrame({"Question": ["Behaviour", "Attack signals", "Progression", "Final verdict"],
                                "Answer": [f"<b>{unusual}</b>", chips(row.b_stages),
                                           f"<b>{len(row.c_stages)} of 6 stages in attack order</b>",
                                           badge(row.C_predicted_class)]}), strong=("Question",))
    elif i == 4:
        m = data.metrics()
        if "test_id" in m.get("splits", []):
            rows = [{"Detector": data.DETECTOR_NAME[d],
                     "F1 (Test ID)": f'{m["detectors"][d]["splits"]["test_id"]["binary"]["f1"]:.3f}',
                     "False positive rate (Test ID)": f'{m["detectors"][d]["splits"]["test_id"]["binary"]["fpr"]:.1%}'}
                    for d in DETECTORS]
            table(pd.DataFrame(rows), numeric=("F1 (Test ID)", "False positive rate (Test ID)"), strong=("Detector",),
                  highlight=pd.Series([d == data.DEFAULT_DETECTOR for d in DETECTORS]))
    else:
        note("Tip: the <b>What am I looking at?</b> button on each page explains it in plain English, and the theme and "
             "dataset switches live at the bottom of the sidebar.")


def tour() -> None:
    i = st.session_state.setdefault("tour_step", 0)
    title, question, page, text = TOUR[i]
    last = i == len(TOUR) - 1
    header("Learn", "Your 60-second tour", "Show me around: five stops through a complete investigation.")
    stops = len(TOUR) - 1                                     # the final screen is a wrap-up, not a stop
    html('<div class="progress">' + "".join(f'<div class="{"done" if k <= i else ""}"></div>'
                                             for k in range(stops)) + "</div>")
    step = "Tour complete" if last else f"Step {i + 1} of {stops}"
    html(f'<div class="eyebrow">{step} &middot; {title}</div>'
         f'<div class="ptitle" style="font-size:1.6rem">{question}</div><div class="psub">{text}</div>')
    st.write("")
    _preview(i)
    st.write("")
    back, nxt, open_, skip, _ = st.columns([0.9, 1.5, 1.5, 1, 1.6])
    with back:
        if st.button("Back", key="tour_back", width="stretch", disabled=i == 0, icon=":material/chevron_left:"):
            st.session_state.tour_step = i - 1
            st.rerun()
    with nxt:
        if st.button("Start investigating" if last else "Next", key="tour_next", type="primary", width="stretch"):
            st.session_state.tour_step = 0 if last else i + 1
            if last:
                nav.go("intruders")
            st.rerun()
    with open_:
        if not last:
            nav.button(f"Open {title}", page, key=f"tour_open_{i}")
    with skip:
        if st.button("Skip tour", key="tour_skip", width="stretch", type="tertiary"):
            st.session_state.tour_step = 0
            nav.go("dashboard")
    footer()
