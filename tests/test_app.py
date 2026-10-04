"""Smoke tests: the Streamlit entry point and every page render without errors on the generated results."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "metrics.json").exists(),
                                reason="pipeline outputs (results/) not present")
PAGES = [("ui.start", "home"), ("ui.start", "guide"), ("ui.start", "tour"),
         ("ui.investigate", "dashboard"), ("ui.investigate", "intruders"), ("ui.investigate", "stories"),
         ("ui.investigate", "evidence"), ("ui.experiment", "evaluation"), ("ui.experiment", "about")]


def render(module: str, name: str, split: str = "test_id"):
    import importlib

    import streamlit as st

    from ui.theme import inject

    st.session_state.setdefault("split", split)
    inject()
    getattr(importlib.import_module(module), name)()


def test_entry_point_renders():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("module, name", PAGES)
def test_page_renders(module, name):
    at = AppTest.from_function(render, args=(module, name), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("split", ["val", "test_ood"])
def test_investigation_pages_render_on_other_datasets(split):
    for name in ("dashboard", "stories", "evidence"):
        at = AppTest.from_function(render, args=("ui.investigate", name, split), default_timeout=300)
        at.run()
        assert not at.exception, (split, name, [e.value for e in at.exception])


def test_tour_steps_through_every_step():
    at = AppTest.from_function(render, args=("ui.start", "tour"), default_timeout=300)
    at.run()
    for step in range(1, 6):
        at.button(key="tour_next").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.session_state.tour_step == step
