"""Reusable, theme-aware building blocks. HTML is rendered with st.markdown and coloured only via CSS variables."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from intruder.contracts import STAGES

from . import data, nav
from .theme import VERDICT_TONE, palette, verdict_color

STAGE_SHORT = {"RECON": "Recon", "CRED_ATTACK": "Credential", "COMPROMISE": "Compromise", "PRIV_ESC": "Privilege",
               "PERSISTENCE": "Persistence", "EXFIL": "Exfiltration"}


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def header(eyebrow: str, title: str, subtitle: str = "", help_md: str | None = None) -> None:
    left, right = st.columns([4.1, 1.65], vertical_alignment="bottom")
    with left:
        html(f'<div class="eyebrow">{eyebrow}</div><div class="ptitle">{title}</div>'
             + (f'<div class="psub">{subtitle}</div>' if subtitle else ""))
    if help_md:
        with right, st.container(key="pagehelp"):
            with st.popover("What am I looking at?", icon=":material/help:", width="stretch"):
                st.markdown(help_md)


def section(title: str, sub: str = "") -> None:
    html(f'<div class="stitle">{title}</div>' + (f'<div class="ssub">{sub}</div>' if sub else '<div style="height:.6rem"></div>'))


def note(body: str, tone: str = "") -> None:
    html(f'<div class="note {tone}">{body}</div>')


def badge(verdict: str) -> str:
    tone = VERDICT_TONE.get(verdict, "neutral")
    return f'<span class="badge {tone}"><i></i>{escape(verdict)}</span>'


STAGE_CODE = {"RECON": "REC", "CRED_ATTACK": "CRD", "COMPROMISE": "CMP", "PRIV_ESC": "PRV", "PERSISTENCE": "PER",
              "EXFIL": "EXF"}


def strip(stages: list[str]) -> str:
    """Fixed-width six-segment stage strip: filled segments are observed stages (full names on hover)."""
    p = palette()
    cells = "".join(
        f'<span class="ss{" on" if s in stages else ""}" style="--c:{p["stage"][s]}" '
        f'title="{data.STAGE_NAME[s]}: {"observed" if s in stages else "not observed"}">{STAGE_CODE[s]}</span>'
        for s in STAGES)
    return f'<div class="strip" aria-label="Stages observed: {", ".join(data.STAGE_NAME[s] for s in stages) or "none"}">{cells}</div>'


def strip_legend() -> str:
    return ('<p class="small muted" style="margin:.5rem 0 0 0">Attack activity: '
            + " &middot; ".join(f"<b>{STAGE_CODE[s]}</b> {data.STAGE_NAME[s]}" for s in STAGES)
            + ". Filled = observed.</p>")


def chips(stages: list[str]) -> str:
    if not stages:
        return '<span class="none">No attack signals</span>'
    p = palette()
    return '<div class="chips">' + "".join(
        f'<span class="chip" style="--c:{p["stage"][s]}"><i></i>{STAGE_SHORT[s]}</span>' for s in stages) + "</div>"


def risk(value: float, verdict: str) -> str:
    v = max(0.0, min(100.0, float(value)))
    return (f'<div class="risk"><b>{v:.0f}</b><div class="bar"><span style="width:{v}%;--tone:{verdict_color(verdict)}">'
            f'</span></div></div>')


def kpis(items: list[tuple[str, str, str, str]]) -> None:
    """items: (label, value, caption, tone colour)."""
    html('<div class="kpis">' + "".join(
        f'<div class="kpi" style="--tone:{c}"><div class="l"><i></i>{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for l, v, s, c in items) + "</div>")


def table(df: pd.DataFrame, numeric: tuple[str, ...] = (), strong: tuple[str, ...] = (), highlight=None,
          nowrap: bool = False) -> None:
    """Theme-aware HTML table. `highlight` is an optional boolean Series aligned with df rows; `nowrap` keeps dense
    log-style rows on one line (the table scrolls sideways instead)."""
    head = "".join(f'<th class="{"num" if c in numeric else ""}">{escape(str(c))}</th>' for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        hl = highlight is not None and bool(highlight.iloc[i])
        cells = "".join(
            f'<td class="{" ".join(x for x in ("num" if c in numeric else "", "strong" if c in strong else "") if x)}">'
            f'{r[c] if isinstance(r[c], str) and r[c].startswith("<") else escape(str(r[c]))}</td>' for c in df.columns)
        rows.append(f'<tr class="{"hl" if hl else ""}">{cells}</tr>')
    html(f'<div class="tablewrap"><table class="t{" nowrap" if nowrap else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def band(row: pd.Series) -> None:
    """Summary band for the entity under investigation (verdict from Attack Progression)."""
    html(f'''<div class="band">
      <div class="who"><div class="eyebrow">Entity</div><b>{escape(row.entity)}</b><span>{data.fmt_day(row.day)} &middot;
        {data.SPLITS[data.current_split()]}</span></div>
      <div class="f"><div class="l">Verdict</div><div class="v">{badge(row.C_predicted_class)}</div></div>
      <div class="f"><div class="l">Risk</div><div class="v">{row.C_risk_score:.0f}<small> / 100</small></div></div>
      <div class="f"><div class="l">Progression</div><div class="v">{row.progression_score:.0f}<small> / 100 &middot;
        {len(row.c_stages)} of 6 stages in order</small></div></div>
      <div class="f"><div class="l">Detection latency</div><div class="v">{data.fmt_duration(row.C_latency_s)}</div></div>
    </div>''')


def progress6(sig: pd.DataFrame, chain: list[str], verdict: str) -> None:
    """Six stages with observed / out-of-order / not observed, and an 'Attack stopped' marker for failed attacks."""
    p = palette()
    first = sig.sort_values("ts").groupby("stage").ts.first() if len(sig) else pd.Series(dtype="datetime64[ns]")
    seen = [s for s in STAGES if s in first.index]
    stop_at = None
    if verdict == "Failed attack" and "COMPROMISE" not in seen:
        stop_at = "COMPROMISE"
    cells = []
    for i, stage in enumerate(STAGES, 1):
        if stage in first.index:
            in_chain = stage in chain
            cls = "s6 on" + ("" if in_chain else " out")
            cells.append(f'<div class="{cls}" style="--c:{p["stage"][stage]}"><div class="dot">&#10003;</div>'
                         f'<div class="nm">{data.STAGE_NAME[stage]}</div><div class="tm">{data.fmt_time(first[stage])}</div>'
                         f'<div class="st">{"In sequence" if in_chain else "Out of order"}</div></div>')
        elif stage == stop_at:
            cells.append('<div class="s6 stop"><div class="dot">&#10005;</div><div class="nm">Attack stopped</div>'
                         f'<div class="tm">no {data.STAGE_NAME[stage].lower()}</div><div class="st">Not reached</div></div>')
        else:
            cells.append(f'<div class="s6"><div class="dot">&mdash;</div><div class="nm">{data.STAGE_NAME[stage]}</div>'
                         f'<div class="tm">not observed</div></div>')
    html(f'<div class="steps6">{"".join(cells)}</div>')


def timeline(split: str, entity: str, sig: pd.DataFrame, show_windows: bool = True) -> None:
    """Human-facing detection timeline; the five-minute processing window is one click away."""
    p = palette()
    items = []
    for e in data.episodes(sig):
        n = e["count"]
        span = data.fmt_hm(e["start"]) if data.fmt_hm(e["start"]) == data.fmt_hm(e["end"]) else \
            f'{data.fmt_hm(e["start"])}&ndash;{data.fmt_hm(e["end"])}'
        body = (f'<div>First signal {data.fmt_time(e["start"])}, last {data.fmt_time(e["end"])} &middot; {n} '
                f'signal{"s" if n != 1 else ""}. Latest evidence: <i>{escape(str(e["evidence"]))}</i></div>')
        if show_windows:
            w0 = pd.Timestamp(e["start"]).floor("5min")
            prof = data.window_profile(split, entity, w0)
            body += (f'<div style="margin-top:.5rem">Processed in window <b>{w0:%H:%M}&ndash;'
                     f'{w0 + pd.Timedelta(minutes=5):%H:%M} UTC</b></div>')
            if prof is not None:
                keys = [k for k in ("failed_logins", "distinct_ports", "denied_connections", "successful_logins",
                                    "sudo_elevated", "new_accounts", "outbound_external_bytes", "event_count")
                        if float(prof[k]) != 0][:6]
                body += '<div class="kv">' + "".join(
                    f'<div>{data.FEATURE_LABEL[k]}<b>{data.feature_value(k, float(prof[k]))}</b></div>' for k in keys) + "</div>"
        items.append(f'<div class="tle" style="--c:{p["stage"][e["stage"]]}"><details><summary>'
                     f'<span class="time">{span}</span><span class="what"><b>{data.STAGE_NAME[e["stage"]]}</b>'
                     f'<span>{data.STAGE_TEXT[e["stage"]]}</span></span><span class="more">Details</span></summary>'
                     f'<div class="body">{body}</div></details></div>')
    html(f'<div class="tl">{"".join(items)}</div>')


def entity_table(view: pd.DataFrame, det: str, key: str, page_size: int = 12, show_progression: bool = True) -> None:
    """Professional entity table built from Streamlit rows so each row has a real 'Investigate' action."""
    s = data.SHORT[det]
    total = len(view)
    if total == 0:
        note("No entities match these filters. Widen the verdict or risk filters to see more.", "warn")
        return
    pages = max(1, -(-total // page_size))
    page = min(st.session_state.get(f"{key}_page", 0), pages - 1)
    chunk = view.iloc[page * page_size:(page + 1) * page_size]
    widths = [2.0, 1.75, 1.45, 2.65] + ([1.3] if show_progression else []) + [1.3, 1.55]
    labels = ["Entity", "Risk", "Verdict", "Attack activity"] + (["Progression"] if show_progression else []) + \
             ["Detection latency", ""]
    with st.container(key=f"rowtable_{key}"):
        with st.container(key=f"thead_{key}"):
            for col, label in zip(st.columns(widths, vertical_alignment="center"), labels):
                col.markdown(f"<p>{label}</p>", unsafe_allow_html=True)
        for i, r in enumerate(chunk.itertuples()):
            verdict = getattr(r, f"{s}_predicted_class")
            stages = r.c_stages if s == "C" else r.b_stages
            with st.container(key=f"erow_{key}_{page}_{i}"):
                cols = st.columns(widths, vertical_alignment="center")
                cols[0].markdown(f'<div class="ent"><b>{escape(r.entity)}</b><span>{data.fmt_day(r.day)}</span></div>',
                                 unsafe_allow_html=True)
                cols[1].markdown(risk(getattr(r, f"{s}_risk_score"), verdict), unsafe_allow_html=True)
                cols[2].markdown(badge(verdict), unsafe_allow_html=True)
                cols[3].markdown(strip(stages), unsafe_allow_html=True)
                k = 4
                if show_progression:
                    cols[4].markdown(f'<span style="font-weight:600;color:var(--text)">{r.progression_score:.0f}</span>'
                                     f'<span class="muted small"> · {len(r.c_stages)}/6</span>', unsafe_allow_html=True)
                    k = 5
                cols[k].markdown(f'<span class="small" style="color:var(--text2)">'
                                 f'{data.fmt_duration(getattr(r, f"{s}_latency_s"))}</span>', unsafe_allow_html=True)
                if cols[k + 1].button("Investigate", key=f"open_{key}_{page}_{i}", type="tertiary",
                                      icon=":material/arrow_forward:"):
                    data.set_focus(r.entity, r.day)
                    nav.go("stories")
    html(strip_legend())
    if pages > 1:
        a, b, c = st.columns([1, 2, 1], vertical_alignment="center")
        if a.button("Previous", key=f"{key}_prev", disabled=page == 0, icon=":material/chevron_left:"):
            st.session_state[f"{key}_page"] = page - 1
            st.rerun()
        b.markdown(f'<p class="muted small" style="text-align:center">Showing {page * page_size + 1}–'
                   f'{min(total, (page + 1) * page_size)} of {total:,} &middot; page {page + 1} of {pages}</p>',
                   unsafe_allow_html=True)
        if c.button("Next", key=f"{key}_next", disabled=page >= pages - 1, icon=":material/chevron_right:"):
            st.session_state[f"{key}_page"] = page + 1
            st.rerun()


def entity_picker(table: pd.DataFrame, key: str) -> pd.Series:
    """Choose an entity-day among those flagged by any detector, ranked by Attack Progression risk."""
    focus = data.focus_row(table)
    pool = table[table.any_alert]
    if not ((pool.entity == focus.entity) & (pool.day == focus.day)).any():
        pool = pd.concat([table[(table.entity == focus.entity) & (table.day == focus.day)], pool])
    labels = {(r.entity, r.day): f"{r.entity}  ·  {data.fmt_day(r.day)}  ·  {r.C_predicted_class}  ·  risk "
                                 f"{r.C_risk_score:.0f}" for r in pool.itertuples()}
    # Options are (entity, day) pairs, not positions, so the choice stays correct when the pool changes; the
    # callback updates the focus before the next run so the sidebar shows the new entity straight away.
    st.session_state[key] = (focus.entity, focus.day)
    entity, day = st.selectbox("Entity", list(labels), format_func=labels.get, key=key,
                               on_change=lambda: data.set_focus(*st.session_state[key]),
                               help="Entities flagged by at least one detector, ranked by Attack Progression risk.")
    return pool[(pool.entity == entity) & (pool.day == day)].iloc[0]


def footer() -> None:
    html('<div class="footer"><span>Find the Intruder &middot; progression-aware ML intrusion detection</span>'
         '<span>Research prototype &middot; synthetic data &middot; offline analysis &middot; no automated response</span></div>')
