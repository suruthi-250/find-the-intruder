"""Design system: two deliberate themes (light and dark) exposed as CSS variables, plus shared component styles.

Every colour used by the UI comes from the active palette, so switching theme never leaves a component behind.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

LIGHT = {
    "bg": "#F5F7FA", "surface": "#FFFFFF", "surface2": "#F8FAFC", "elev": "#FFFFFF", "sidebar": "#FFFFFF",
    "border": "#E4E7EC", "border2": "#D0D5DD", "text": "#0B1220", "text2": "#344054", "muted": "#667085",
    "faint": "#98A2B3", "accent": "#2563EB", "accent2": "#0E7490", "accent_soft": "#EEF4FF", "accent_line": "#C7D7FE",
    "ok": "#15803D", "ok_soft": "#ECFDF3", "warn": "#A15C07", "warn_soft": "#FEF7E6", "high": "#C2410C",
    "high_soft": "#FFF3EB", "crit": "#B42318", "crit_soft": "#FEF2F2", "shadow": "0 1px 2px rgba(16,24,40,.05)",
    "shadow2": "0 12px 32px -12px rgba(16,24,40,.18)", "hero1": "#FFFFFF", "hero2": "#EEF4FF", "hero3": "#E6F6FA",
    "glass": "rgba(255,255,255,.72)", "grid": "rgba(37,99,235,.06)", "on_accent": "#FFFFFF",
    "stage": {"RECON": "#0E7490", "CRED_ATTACK": "#4F46E5", "COMPROMISE": "#B42318", "PRIV_ESC": "#C2410C",
              "PERSISTENCE": "#7C3AED", "EXFIL": "#BE123C"},
    "detector": {"A_anomaly_only": "#98A2B3", "B_stage_count": "#60A5FA", "C_progression": "#2563EB"},
}
DARK = {
    "bg": "#080B12", "surface": "#0F1521", "surface2": "#121A28", "elev": "#172033", "sidebar": "#0A0E18",
    "border": "#222B3B", "border2": "#2E3950", "text": "#E7ECF5", "text2": "#B4BED0", "muted": "#8391A8",
    "faint": "#5D6A80", "accent": "#38BDF8", "accent2": "#22D3EE", "accent_soft": "rgba(56,189,248,.12)",
    "accent_line": "rgba(56,189,248,.35)", "ok": "#34D399", "ok_soft": "rgba(52,211,153,.12)", "warn": "#FBBF24",
    "warn_soft": "rgba(251,191,36,.12)", "high": "#FB923C", "high_soft": "rgba(251,146,60,.13)", "crit": "#F87171",
    "crit_soft": "rgba(248,113,113,.13)", "shadow": "0 1px 2px rgba(0,0,0,.4)",
    "shadow2": "0 18px 40px -18px rgba(0,0,0,.8)", "hero1": "#0B1020", "hero2": "#111A33", "hero3": "#0B1A26",
    "glass": "rgba(23,32,51,.62)", "grid": "rgba(56,189,248,.05)", "on_accent": "#06121F",
    "stage": {"RECON": "#22D3EE", "CRED_ATTACK": "#818CF8", "COMPROMISE": "#F87171", "PRIV_ESC": "#FB923C",
              "PERSISTENCE": "#A78BFA", "EXFIL": "#FB7185"},
    "detector": {"A_anomaly_only": "#64748B", "B_stage_count": "#60A5FA", "C_progression": "#38BDF8"},
}
VERDICT_TONE = {"Intrusion": "crit", "Failed attack": "high", "Suspicious": "warn", "Normal": "neutral"}


def mode() -> str:
    return st.session_state.get("theme", "light")


def palette() -> dict:
    return DARK if mode() == "dark" else LIGHT


def verdict_color(verdict: str) -> str:
    p = palette()
    tone = VERDICT_TONE.get(verdict, "neutral")
    return p["faint"] if tone == "neutral" else p[tone]


BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, textarea, select, label,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * , [data-testid="stWidgetLabel"] * {
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1320px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stMarkdownContainer"] { color:var(--text); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--text2); line-height:1.6; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
[data-testid="stWidgetLabel"] p { color:var(--text2) !important; font-weight:500; font-size:.84rem; }
a { color:var(--accent); }

/* sidebar */
[data-testid="stSidebar"] { background:var(--sidebar) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background:var(--sidebar); }
[data-testid="stSidebarNavItems"] a { border-radius:8px; }
[data-testid="stSidebarNavItems"] a span { color:var(--text2) !important; font-weight:500; font-size:.9rem; }
[data-testid="stSidebarNavItems"] a:hover { background:var(--surface2); }
[data-testid="stSidebarNavItems"] a[aria-current="page"] { background:var(--accent-soft); }
[data-testid="stSidebarNavItems"] a[aria-current="page"] span { color:var(--accent) !important; font-weight:600; }
[data-testid="stSidebarNavItems"] [data-testid="stIconMaterial"] { color:var(--muted); }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] * {
  text-transform:uppercase; letter-spacing:.14em; font-size:.66rem !important; font-weight:600; color:var(--faint) !important; }
[data-testid="stSidebarHeader"] { background:var(--sidebar); }
[data-testid="stSidebar"] hr { border-color:var(--border); }
[data-testid="stSidebarNavSeparator"] { border-color:var(--border) !important; margin-bottom:1.1rem; }

/* inputs, menus, toggles */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {
  background:var(--surface) !important; border-color:var(--border2) !important; border-radius:9px !important; }
[data-baseweb="select"] *, [data-baseweb="input"] input, [data-baseweb="base-input"] input { color:var(--text) !important; }
[data-baseweb="select"] svg { fill:var(--muted) !important; }
[data-testid="stSelectbox"] div[role="group"], [data-testid="stMultiSelect"] div[role="group"],
[data-testid="stTextInput"] div[role="group"], [data-testid="stTextInput"] input {
  background:var(--surface) !important; border-color:var(--border2) !important; color:var(--text) !important; }
[data-testid="stSelectbox"] div[role="group"] *, [data-testid="stMultiSelect"] div[role="group"] *,
input[role="combobox"], [data-testid="stTextInput"] input { color:var(--text) !important; }
input::placeholder { color:var(--faint) !important; }
[data-testid="stCheckbox"] label > div:not([data-testid]) { background:var(--border2) !important; }
[data-testid="stCheckbox"] label input:checked + div { background:var(--accent) !important; }
[data-testid="stRadio"] label p, [data-testid="stRadio"] label div { color:var(--text2) !important; }
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] { background:var(--elev) !important; }
ul[role="listbox"] li, [role="option"] { color:var(--text) !important; }
ul[role="listbox"] li:hover, [role="option"][aria-selected="true"] { background:var(--accent-soft) !important; }
[data-testid$="VirtualDropdown"] { background:var(--elev) !important; border:1px solid var(--border2) !important;
  border-radius:10px !important; box-shadow:var(--shadow2) !important; color:var(--text) !important; }
[data-testid$="VirtualDropdown"] [role="option"] { border-radius:6px; }
[data-testid$="VirtualDropdown"] [role="option"][data-hovered], [data-testid$="VirtualDropdown"] [role="option"][data-focused] {
  background:var(--surface2) !important; }
[data-testid$="VirtualDropdown"] [role="option"][aria-selected="true"] { background:var(--accent-soft) !important; }
[data-testid$="VirtualDropdown"] [role="option"][aria-selected="true"] * { color:var(--accent) !important; }
[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] [data-tag] {
  background:var(--accent-soft) !important; border:1px solid var(--accent-line) !important; border-radius:6px !important; }
[data-baseweb="tag"] span, [data-testid="stMultiSelectTagsContainer"] [data-tag] span,
[data-testid="stMultiSelectTagsContainer"] [data-tag] button { color:var(--accent) !important; font-weight:500; }
[data-testid="stTextInputRootElement"] { background:var(--surface) !important; border-color:var(--border2) !important;
  border-radius:9px !important; }
[data-testid="stTextInputRootElement"]:focus-within { border-color:var(--accent) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] { color:var(--muted) !important; }
[data-testid="stButtonGroup"] button { background:var(--surface) !important; border-color:var(--border2) !important; }
[data-testid="stButtonGroup"] button p, [data-testid="stButtonGroup"] button [data-testid="stIconMaterial"] {
  color:var(--text2) !important; }
[data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] * { color:var(--muted) !important; }
[data-testid="stTooltipIcon"] svg { stroke:var(--muted) !important; }
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarHeader"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] { color:var(--muted) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background:var(--accent-soft) !important; border-color:var(--accent) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] p, [data-testid="stButtonGroup"] button[aria-pressed="true"] p,
[data-testid="stButtonGroup"] button[aria-checked="true"] [data-testid="stIconMaterial"] { color:var(--accent) !important; font-weight:600; }
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * { background:var(--elev) !important; color:var(--text) !important; }

/* buttons */
div[data-testid="stButton"] button, div[data-testid="stPopover"] > div > button {
  border-radius:9px; font-weight:600; border:1px solid var(--border2); background:var(--surface); color:var(--text);
  transition:all .15s ease; }
div[data-testid="stButton"] button p, div[data-testid="stPopover"] button p { color:var(--text) !important; font-weight:600; }
div[data-testid="stButton"] button:hover { border-color:var(--accent); background:var(--surface2); }
div[data-testid="stButton"] button[kind="primary"] { background:var(--accent); border-color:var(--accent); }
div[data-testid="stButton"] button[kind="primary"] p { color:var(--on-accent) !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { filter:brightness(1.07); background:var(--accent); }
div[data-testid="stButton"] button[kind="tertiary"] { border:none; background:transparent; }
div[data-testid="stButton"] button[kind="tertiary"] p { color:var(--accent) !important; }
div[data-testid="stButton"] button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
div[data-testid="stButton"] button:disabled { opacity:.45; }

/* expanders and popovers */
[data-testid="stExpander"] details { background:var(--surface); border:1px solid var(--border) !important; border-radius:12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:var(--text) !important; font-weight:600; }
[data-testid="stExpander"] svg { fill:var(--muted); color:var(--muted); }
[data-testid="stPopoverBody"] { background:var(--elev) !important; border:1px solid var(--border) !important; }
[data-testid="stPopoverBody"] * { color:var(--text2); }
[data-testid="stPopoverBody"] strong { color:var(--text); }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color:var(--muted) !important; background:transparent !important; }
[data-testid="stSkeleton"] { background:var(--surface2) !important; }
[data-testid="stAlert"] { background:var(--surface2) !important; }

/* ---------------- typography ---------------- */
.eyebrow { font-size:.7rem; font-weight:600; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); }
.ptitle { font-size:2.05rem; font-weight:700; letter-spacing:-.02em; color:var(--text); line-height:1.15; margin:.35rem 0 0 0; }
.psub { font-size:1rem; color:var(--muted); margin-top:.35rem; }
.stitle { font-size:1.12rem; font-weight:650; color:var(--text); letter-spacing:-.01em; margin:2.1rem 0 .2rem 0; }
.ssub { font-size:.86rem; color:var(--muted); margin-bottom:.9rem; }
.muted { color:var(--muted) !important; }
.small { font-size:.82rem; }
.mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size:.78rem; }
.divider { height:1px; background:var(--border); margin:1.8rem 0; }

/* ---------------- surfaces ---------------- */
.panel { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.2rem 1.35rem;
         box-shadow:var(--shadow); }
.panel h4 { margin:0 0 .35rem 0; font-size:.98rem; font-weight:650; color:var(--text); }
.panel p { margin:.2rem 0; color:var(--text2); font-size:.9rem; }
.panel.accent { border-color:var(--accent-line); box-shadow:0 0 0 3px var(--accent-soft); }
.note { background:var(--surface2); border:1px solid var(--border); border-left:3px solid var(--accent); border-radius:10px;
        padding:.85rem 1.05rem; color:var(--text2); font-size:.9rem; line-height:1.6; }
.note b { color:var(--text); }
.note.warn { border-left-color:var(--warn); }
.note.crit { border-left-color:var(--crit); }
.note.ok { border-left-color:var(--ok); }

/* KPIs */
.kpis { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:.9rem; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1rem 1.15rem; box-shadow:var(--shadow);
       position:relative; overflow:hidden; }
.kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--tone); }
.kpi .l { display:flex; align-items:center; gap:.45rem; font-size:.74rem; font-weight:600; letter-spacing:.06em;
          text-transform:uppercase; color:var(--muted); }
.kpi .l i { width:8px; height:8px; border-radius:50%; background:var(--tone); }
.kpi .v { font-size:2.05rem; font-weight:700; color:var(--text); letter-spacing:-.02em; margin-top:.35rem;
          font-variant-numeric:tabular-nums; }
.kpi .s { font-size:.8rem; color:var(--muted); margin-top:.1rem; }

/* badges and chips */
.badge { display:inline-flex; align-items:center; gap:.38rem; font-size:.76rem; font-weight:600; padding:.18rem .58rem;
         border-radius:6px; white-space:nowrap; border:1px solid transparent; }
.badge i { width:7px; height:7px; border-radius:50%; background:currentColor; }
.badge.crit { color:var(--crit); background:var(--crit-soft); border-color:color-mix(in srgb, var(--crit) 30%, transparent); }
.badge.high { color:var(--high); background:var(--high-soft); border-color:color-mix(in srgb, var(--high) 30%, transparent); }
.badge.warn { color:var(--warn); background:var(--warn-soft); border-color:color-mix(in srgb, var(--warn) 30%, transparent); }
.badge.neutral { color:var(--muted); background:var(--surface2); border-color:var(--border); }
.badge.ok { color:var(--ok); background:var(--ok-soft); }
.badge.info { color:var(--accent); background:var(--accent-soft); }
.chips { display:flex; flex-wrap:wrap; gap:.3rem; }
.chip { display:inline-flex; align-items:center; gap:.3rem; font-size:.72rem; font-weight:600; padding:.14rem .45rem;
        border-radius:5px; color:var(--text2); background:var(--surface2); border:1px solid var(--border); white-space:nowrap; }
.chip i { width:6px; height:6px; border-radius:2px; background:var(--c); }
.none { color:var(--faint); font-size:.82rem; }
.strip { display:flex; gap:3px; }
.ss { font-size:.6rem; font-weight:700; letter-spacing:.04em; width:34px; text-align:center; padding:.22rem 0; border-radius:4px;
      color:var(--faint); background:transparent; border:1px dashed var(--border2); cursor:default; }
.ss.on { color:var(--c); background:color-mix(in srgb, var(--c) 14%, transparent); border:1px solid color-mix(in srgb, var(--c) 45%, transparent); }

/* risk bar */
.risk { display:flex; align-items:center; gap:.55rem; }
.risk b { font-variant-numeric:tabular-nums; color:var(--text); font-weight:650; min-width:1.6rem; font-size:.92rem; }
.risk .bar { flex:1; height:6px; border-radius:6px; background:var(--surface2); border:1px solid var(--border); overflow:hidden;
             min-width:48px; }
.risk .bar span { display:block; height:100%; border-radius:6px; background:var(--tone); }

/* tables */
table.t { width:100%; border-collapse:separate; border-spacing:0; font-size:.86rem; background:var(--surface);
          border:1px solid var(--border); border-radius:12px; overflow:hidden; }
table.t th { text-align:left; font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
             background:var(--surface2); padding:.6rem .8rem; border-bottom:1px solid var(--border); white-space:nowrap; }
table.t td { padding:.62rem .8rem; border-bottom:1px solid var(--border); color:var(--text2); vertical-align:middle; }
table.t tr:last-child td { border-bottom:none; }
table.t tr:hover td { background:var(--surface2); }
table.t td.num, table.t th.num { text-align:right; font-variant-numeric:tabular-nums; }
table.t td.strong { color:var(--text); font-weight:600; }
table.t tr.hl td { background:var(--accent-soft); }
table.t tr.hl td:first-child { box-shadow:inset 3px 0 0 var(--accent); }
.tablewrap { overflow-x:auto; border-radius:12px; }
table.t.nowrap td, table.t.nowrap th { white-space:nowrap; }
table.t.nowrap td { padding:.5rem .8rem; font-size:.82rem; }

/* streamlit-built row tables */
[class*="st-key-thead"] { border-bottom:1px solid var(--border); padding:.1rem .6rem .35rem .6rem; }
[class*="st-key-thead"] p { margin:0; font-size:.68rem !important; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
                            color:var(--muted) !important; }
[class*="st-key-erow"] { border-bottom:1px solid var(--border); padding:.55rem .6rem; border-radius:8px; transition:background .12s; }
[class*="st-key-erow"]:hover { background:var(--surface2); }
[class*="st-key-erow"] p { margin:0; }
.ent b { color:var(--text); font-weight:600; font-size:.9rem; }
.ent span { display:block; color:var(--muted); font-size:.76rem; }
[class*="st-key-rowtable"] { background:var(--surface); border:1px solid var(--border); border-radius:12px;
                             padding:.7rem .5rem .4rem .5rem; box-shadow:var(--shadow); gap:.1rem !important; }
[class*="st-key-erow"] .ent span { margin-top:.1rem; }
[class*="st-key-erow"] div[data-testid="stButton"] button { padding:.2rem .4rem; min-height:0; }

/* help popover trigger */
.st-key-pagehelp button { border-radius:999px !important; font-size:.82rem !important; }

/* stage progress (six steps) */
.steps6 { display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:0; position:relative; margin:1rem 0 .4rem 0; }
.steps6:before { content:""; position:absolute; left:8%; right:8%; top:17px; height:2px; background:var(--border); }
.s6 { text-align:center; position:relative; padding:0 .25rem; }
.s6 .dot { width:36px; height:36px; border-radius:50%; margin:0 auto; display:flex; align-items:center; justify-content:center;
           font-size:.8rem; font-weight:700; position:relative; z-index:1; background:var(--bg); color:var(--faint);
           border:2px dashed var(--border2); }
.s6.on .dot { background:var(--c); color:#fff; border:2px solid var(--c); }
.s6.on.out .dot { background:var(--surface); color:var(--c); }
.s6 .nm { font-size:.82rem; font-weight:600; color:var(--text); margin-top:.5rem; }
.s6:not(.on) .nm { color:var(--faint); }
.s6 .tm { font-size:.74rem; color:var(--muted); margin-top:.1rem; font-variant-numeric:tabular-nums; }
.s6 .st { font-size:.64rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-top:.3rem; color:var(--accent); }
.s6.out .st { color:var(--high); }
.s6.stop .dot { border-color:var(--crit); color:var(--crit); border-style:solid; background:var(--crit-soft); }
.s6.stop .st { color:var(--crit); }
.s6.stop .nm { color:var(--crit); }

/* detection timeline */
.tl { position:relative; padding-left:1.6rem; }
.tl:before { content:""; position:absolute; left:.45rem; top:.4rem; bottom:.4rem; width:2px; background:var(--border); }
.tle { position:relative; margin-bottom:.8rem; }
.tle:before { content:""; position:absolute; left:-1.42rem; top:.95rem; width:12px; height:12px; border-radius:50%;
              background:var(--c); box-shadow:0 0 0 4px var(--bg); }
.tle details { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.75rem 1rem; }
.tle summary { list-style:none; cursor:pointer; display:flex; gap:1rem; align-items:baseline; }
.tle summary::-webkit-details-marker { display:none; }
.tle .time { font-weight:700; color:var(--text); font-variant-numeric:tabular-nums; min-width:4.6rem; }
.tle .what { flex:1; }
.tle .what b { display:block; color:var(--text); font-weight:600; font-size:.92rem; }
.tle .what span { color:var(--text2); font-size:.86rem; }
.tle .more { font-size:.74rem; color:var(--accent); white-space:nowrap; }
.tle details[open] .more { color:var(--muted); }
.tle .body { margin-top:.7rem; padding-top:.7rem; border-top:1px solid var(--border); font-size:.84rem; color:var(--text2); }
.tle .body .kv { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.35rem 1rem; margin-top:.5rem; }
.tle .body .kv div { color:var(--muted); font-size:.78rem; }
.tle .body .kv b { color:var(--text); font-weight:600; margin-left:.3rem; }
.stopcard { border:1px dashed var(--crit); background:var(--crit-soft); border-radius:12px; padding:.85rem 1rem; color:var(--text2);
            font-size:.88rem; margin-left:0; }
.stopcard b { color:var(--crit); letter-spacing:.06em; }

/* summary band */
.band { display:flex; flex-wrap:wrap; gap:1.8rem; align-items:flex-end; background:var(--surface); border:1px solid var(--border);
        border-radius:14px; padding:1.1rem 1.35rem; box-shadow:var(--shadow); }
.band .who { flex:1 1 220px; }
.band .who b { display:block; font-size:1.45rem; font-weight:700; color:var(--text); letter-spacing:-.01em; }
.band .who span { color:var(--muted); font-size:.84rem; }
.band .f .l { font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.band .f .v { font-size:1.2rem; font-weight:650; color:var(--text); margin-top:.25rem; font-variant-numeric:tabular-nums; }
.band .f .v small { font-size:.78rem; color:var(--muted); font-weight:500; }

/* question flow: unusual != malicious */
.qflow { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:.7rem; }
.qf { position:relative; background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1rem 1.1rem;
      box-shadow:var(--shadow); }
.qf:not(:last-child):after { content:"\\2192"; position:absolute; right:-.62rem; top:42%; color:var(--faint); z-index:1; }
.qf .n { font-size:.66rem; font-weight:700; letter-spacing:.14em; color:var(--muted); }
.qf .q { font-weight:650; color:var(--text); margin-top:.35rem; font-size:.95rem; line-height:1.3; }
.qf .c { font-size:1.6rem; font-weight:700; color:var(--text); margin-top:.6rem; font-variant-numeric:tabular-nums; }
.qf .d { font-size:.76rem; color:var(--muted); }
.qf.last { border-color:color-mix(in srgb, var(--crit) 40%, var(--border)); background:linear-gradient(180deg, var(--crit-soft), var(--surface)); }

/* hero */
.hero { position:relative; overflow:hidden; border-radius:20px; border:1px solid var(--border);
        background:radial-gradient(ellipse at 85% 10%, var(--hero2) 0%, transparent 55%),
                   radial-gradient(ellipse at 70% 100%, var(--hero3) 0%, transparent 50%), var(--hero1);
        padding:2.6rem 2.6rem 2.4rem 2.6rem; box-shadow:var(--shadow2); }
.hero:before { content:""; position:absolute; inset:0; pointer-events:none;
               background-image:linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px);
               background-size:36px 36px; mask-image:linear-gradient(90deg, transparent 0%, #000 55%); }
.hero .wrap { position:relative; display:grid; grid-template-columns:1.12fr 1fr; gap:2.4rem; align-items:center; }
.hero .brand { font-size:.72rem; font-weight:700; letter-spacing:.2em; text-transform:uppercase; color:var(--accent); }
.hero .brand span { color:var(--muted); font-weight:600; letter-spacing:.12em; }
.hero .h1 { font-size:2.85rem; line-height:1.08; font-weight:700; letter-spacing:-.025em; color:var(--text);
            margin:.9rem 0 1rem 0; }
.hero .h1 em { font-style:normal; color:var(--accent); }
.hero .lead { color:var(--text2); font-size:1.02rem; line-height:1.65; max-width:560px; }
.stack { display:flex; flex-direction:column; gap:.5rem; }
.layer { display:flex; align-items:center; gap:.85rem; background:var(--glass); backdrop-filter:blur(10px);
         border:1px solid var(--border); border-radius:12px; padding:.7rem .95rem; box-shadow:var(--shadow);
         animation:drift 7s ease-in-out infinite; animation-delay:calc(var(--i) * -1.1s); }
.layer .ix { width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center;
             font-size:.72rem; font-weight:700; color:var(--c); background:color-mix(in srgb, var(--c) 14%, transparent);
             border:1px solid color-mix(in srgb, var(--c) 35%, transparent); }
.layer .tx { flex:1; }
.layer .tx b { display:block; color:var(--text); font-size:.9rem; font-weight:650; }
.layer .tx span { color:var(--muted); font-size:.76rem; }
.layer .vv { font-weight:700; color:var(--text); font-variant-numeric:tabular-nums; font-size:1rem; text-align:right; }
.layer .vv small { display:block; font-size:.66rem; color:var(--muted); font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.layer.final { border-color:color-mix(in srgb, var(--crit) 45%, var(--border)); background:linear-gradient(90deg, var(--crit-soft), var(--glass)); }
.layer:nth-child(2) { margin-left:.8rem; } .layer:nth-child(3) { margin-left:1.6rem; } .layer:nth-child(4) { margin-left:2.4rem; }
.layer:nth-child(5) { margin-left:3.2rem; }
@keyframes drift { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-3px); } }
.calm .layer { animation:none; }
@media (prefers-reduced-motion: reduce) { .layer { animation:none; } }
.st-key-hero_cta { margin-top:-96px; padding-left:2.6rem; position:relative; z-index:3; margin-bottom:60px; }

/* guide */
.qa { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:.8rem; }
.qa .panel .q { font-weight:650; color:var(--text); font-size:1rem; }
.qa .panel .a { color:var(--text2); font-size:.9rem; line-height:1.6; margin-top:.35rem; }
.qa details { margin-top:.6rem; border-top:1px solid var(--border); padding-top:.5rem; }
.qa summary { cursor:pointer; color:var(--accent); font-size:.78rem; font-weight:600; }
.qa details p { color:var(--muted) !important; font-size:.84rem; margin:.4rem 0 0 0; }
.toc { display:flex; flex-wrap:wrap; gap:.4rem; margin:1rem 0 .4rem 0; }
.toc a { font-size:.8rem; color:var(--accent) !important; text-decoration:none; padding:.3rem .65rem; border-radius:999px;
         border:1px solid var(--accent-line); background:var(--accent-soft); }
.grp { font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin:1.8rem 0 .7rem 0; }

/* pipeline diagram */
.pipe { display:grid; grid-template-columns:repeat(var(--per-row, 5), minmax(0, 1fr)); gap:.45rem; align-items:stretch;
        margin-bottom:1.1rem; }
.pipe .p { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.65rem .75rem;
           position:relative; }
@media (max-width: 1100px) { .pipe { grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); } }
.pipe .p .n { font-size:.62rem; font-weight:700; letter-spacing:.12em; color:var(--muted); }
.pipe .p .t { font-size:.84rem; font-weight:600; color:var(--text); margin-top:.15rem; line-height:1.3; }
.pipe .p.hl { border-color:var(--accent-line); background:var(--accent-soft); }
.pipe .p.hl .t { color:var(--accent); }
.pipe .p.out { border-color:color-mix(in srgb, var(--crit) 35%, var(--border)); }
.pipe .p.dashed { border-style:dashed; }
.tagc { display:inline-block; font-size:.62rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--warn);
        background:var(--warn-soft); border:1px solid color-mix(in srgb, var(--warn) 35%, transparent); padding:.18rem .5rem;
        border-radius:999px; margin-left:.6rem; vertical-align:middle; }

/* tour */
.progress { display:flex; gap:.35rem; margin:.6rem 0 1.4rem 0; }
.progress div { flex:1; height:4px; border-radius:4px; background:var(--border); }
.progress div.done { background:var(--accent); }

.footer { margin-top:2.6rem; padding-top:1rem; border-top:1px solid var(--border); color:var(--faint); font-size:.76rem;
          display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem; }
@media (max-width:1100px) {
  .hero .wrap { grid-template-columns:1fr; } .kpis, .qflow { grid-template-columns:repeat(2, minmax(0,1fr)); }
  .qa { grid-template-columns:1fr; } .tle .body .kv { grid-template-columns:repeat(2, minmax(0,1fr)); }
  .st-key-hero_cta { margin-top:0; padding-left:0; margin-bottom:1rem; }
}
"""


def _variables(p: dict) -> str:
    keys = {k: v for k, v in p.items() if isinstance(v, str)}
    body = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in keys.items())
    return f":root, .stApp {{{body}}}"


def inject() -> None:
    p = palette()
    st.markdown(f"<style>{_variables(p)}\n{BASE_CSS}</style>", unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 300) -> go.Figure:
    p = palette()
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=34, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=p["text2"]),
        title_font=dict(size=13, color=p["text"]), legend=dict(orientation="h", y=-0.2, font=dict(size=11, color=p["text2"])),
        hoverlabel=dict(bgcolor=p["elev"], font_color=p["text"], bordercolor=p["border"]),
    )
    fig.update_xaxes(showgrid=False, linecolor=p["border"], tickfont=dict(color=p["muted"]), zeroline=False)
    fig.update_yaxes(gridcolor=p["border"], zeroline=False, linecolor=p["border"], tickfont=dict(color=p["muted"]))
    fig.update_annotations(font=dict(color=p["text"], size=12))
    return fig
