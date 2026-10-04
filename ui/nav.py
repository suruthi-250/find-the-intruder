"""Page registry so any page can send the user to another one."""
from __future__ import annotations

import streamlit as st

PAGES: dict = {}


def register(pages: dict) -> None:
    PAGES.update(pages)


def go(name: str) -> None:
    if name in PAGES:
        st.switch_page(PAGES[name])


def button(label: str, name: str, *, key: str, primary: bool = False) -> None:
    if st.button(label, key=key, type="primary" if primary else "secondary", width="stretch"):
        go(name)
