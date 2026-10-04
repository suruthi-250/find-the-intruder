"""Find the Intruder: progression-aware ML intrusion detection (Streamlit front end).

    streamlit run app.py

Presentation only. Pages live in ui/ and read the outputs written by train.py, predict.py and
`py -m evaluation.run`. Nothing here changes a detector's decision.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui import data, nav
from ui.experiment import about, evaluation
from ui.investigate import dashboard, evidence, intruders, stories
from ui.start import guide, tour
from ui.theme import inject, mode

ASSETS = Path(__file__).resolve().parent / "assets"
THEMES = {"light": ":material/light_mode: Light", "dark": ":material/dark_mode: Dark"}

st.set_page_config(page_title="Find the Intruder", page_icon=str(ASSETS / "mark.svg"), layout="wide",
                   initial_sidebar_state="expanded")

# The theme is kept in a plain session key (widget state does not survive st.switch_page) and mirrored in the
# URL so it also survives a browser refresh.
if "theme" not in st.session_state:
    st.session_state["theme"] = st.query_params.get("theme") if st.query_params.get("theme") in THEMES else "light"
st.session_state["theme_picker"] = st.session_state["theme"]


def _theme_changed() -> None:
    if st.session_state.get("theme_picker") in THEMES:      # segmented controls can be cleared; keep a theme
        st.session_state["theme"] = st.session_state["theme_picker"]


def _sync_theme_url() -> None:
    want = None if st.session_state["theme"] == "light" else st.session_state["theme"]
    if st.query_params.get("theme") != want:
        if want:
            st.query_params["theme"] = want
        else:
            del st.query_params["theme"]


PAGES = {
    "dashboard": st.Page(dashboard, title="Dashboard", icon=":material/space_dashboard:", default=True),
    "intruders": st.Page(intruders, title="Intruders", icon=":material/person_search:", url_path="intruders"),
    "stories": st.Page(stories, title="Attack Stories", icon=":material/timeline:", url_path="attack-stories"),
    "evidence": st.Page(evidence, title="Evidence", icon=":material/fact_check:", url_path="evidence"),
    "evaluation": st.Page(evaluation, title="Evaluation", icon=":material/insights:", url_path="evaluation"),
    "about": st.Page(about, title="About the Model", icon=":material/hub:", url_path="about"),
    "guide": st.Page(guide, title="User Guide", icon=":material/menu_book:", url_path="guide"),
    "tour": st.Page(tour, title="Quick Tour", icon=":material/explore:", url_path="tour"),
}
nav.register(PAGES)
page = st.navigation({
    "Overview": [PAGES["dashboard"]],
    "Investigate": [PAGES["intruders"], PAGES["stories"], PAGES["evidence"]],
    "Evaluate": [PAGES["evaluation"], PAGES["about"]],
    "Learn": [PAGES["guide"], PAGES["tour"]],
})

with st.sidebar:
    st.segmented_control("Theme", list(THEMES), format_func=THEMES.get, key="theme_picker", on_change=_theme_changed,
                         width="stretch")
    available = [s for s in data.SPLITS if (data.RESULTS / s / "window_scores.csv").exists()] or list(data.SPLITS)
    if st.session_state.get("split") not in available:
        st.session_state["split"] = available[0]
    st.session_state["split_picker"] = st.session_state["split"]
    st.selectbox("Dataset", available, format_func=data.SPLITS.get, key="split_picker",
                 on_change=lambda: st.session_state.update(split=st.session_state["split_picker"]),
                 help="Test (ID): same behavioural profile as training. Test (OOD): deliberately shifted benign "
                      "behaviour. Validation: used during development.")
    focus = st.session_state.get("focus")
    if focus and focus[0] == st.session_state.get("split"):
        st.markdown(f"<div style='margin-top:.4rem;font-size:.66rem;font-weight:600;letter-spacing:.12em;"
                    f"text-transform:uppercase;color:var(--faint)'>Investigating</div><div style='font-weight:600;"
                    f"color:var(--text)'>{focus[1]}</div><div style='font-size:.8rem;color:var(--muted)'>"
                    f"{data.fmt_day(focus[2])}</div>", unsafe_allow_html=True)
    st.session_state["motion_picker"] = st.session_state.setdefault("reduce_motion", False)
    st.toggle("Reduce motion", key="motion_picker", help="Turns off the subtle motion in the overview.",
              on_change=lambda: st.session_state.update(reduce_motion=st.session_state["motion_picker"]))

_sync_theme_url()
inject()
st.logo(str(ASSETS / ("logo_dark.svg" if mode() == "dark" else "logo_light.svg")), icon_image=str(ASSETS / "mark.svg"),
        size="large")

if not data.ready():
    st.markdown("### No detection results yet")
    st.markdown("Run the pipeline from the project folder, then reload this page:")
    st.code("py train.py --data data/generated --out models --seed 42\n"
            "py predict.py --data data/generated --model models/isolation_forest.pkl "
            "--splits val test_id test_ood --out results\n"
            "py -m evaluation.run --predictions results/predictions.csv --data data/generated "
            "--splits val test_id test_ood --out results", language="bash")
    st.stop()

page.run()
