"""Clinician dashboard in the spirit of Apple's Human Interface Guidelines.

Clarity, deference and depth: content first, translucent "glass" surfaces,
system typography, grouped lists and a grounded AI assistant alongside the
patient's chart.

Run with:  streamlit run app.py
"""

import base64
import html
import os
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from src import rag_pipeline
from src.fhir_loader import load_fhir_bundle
from src.patient_summary import (
    allergies,
    key_measures,
    medication_list,
    observation_series,
    patient_header,
    problem_list,
    timeline_events,
)

# Number of patients indexed on the first run. Changing it requires deleting
# the storage/ folder so the index is rebuilt with the new set of patients.
PATIENT_LIMIT = int(os.environ.get("PATIENT_LIMIT", 20))
# Questions per browser session; protects the shared API quota of the public demo.
MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS", 25))

SUGGESTED_QUESTIONS = [
    "Summarise the cardiovascular history",
    "Which medications are active?",
    "How has blood pressure changed?",
]

# Apple system colours. The three chart colours pass colour-blind checks
# (CVD ΔE ≥ 18 across all pairs); identity is never carried by colour alone.
BLUE, ORANGE, TEAL = "#007AFF", "#FF9500", "#30B0C7"
TILE_STYLE = {  # tile accent colour and SF-Symbols-like icon
    "Blood pressure": ("#5856D6", "M2 12h4l2-5 4 10 2-5h8"),
    "BMI": ("#AF52DE", "M12 3a2.5 2.5 0 1 1 0 5a2.5 2.5 0 0 1 0-5M7 10h10l-1.5 11h-7z"),
    "Heart rate": ("#FF2D55", "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"),
    "HbA1c": ("#30B0C7", "M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"),
}
UNITS = {"mm[Hg]": "mmHg", "/min": "bpm", "kg/m2": "kg/m²"}
FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif'

st.set_page_config(page_title="Clinical RAG", page_icon="🩺", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --label: #1c1c1e; --secondary: rgba(60,60,67,.62); --tertiary: rgba(60,60,67,.32);
  --separator: rgba(60,60,67,.13); --fill: rgba(118,118,128,.12);
  --glass: rgba(255,255,255,.64); --glass-border: rgba(255,255,255,.8);
  --blue: #007AFF; --red: #D70015; --red-fill: rgba(255,59,48,.12);
  --shadow: 0 1px 1px rgba(0,0,0,.03), 0 12px 32px rgba(0,0,0,.06);
}
/* System typography everywhere except icon fonts and code */
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre):not(.material-symbols-rounded) {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
/* A soft, colourful backdrop gives the glass something to refract */
.stApp {
  background:
    radial-gradient(1100px 620px at 8% -12%, rgba(0,122,255,.13), transparent 62%),
    radial-gradient(900px 560px at 102% -4%, rgba(175,82,222,.10), transparent 60%),
    radial-gradient(900px 700px at 55% 115%, rgba(48,176,199,.10), transparent 60%),
    #f2f2f7;
  color: var(--label);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; max-width: 1480px; }

/* Glass material */
.glass, .st-key-assistant, [class*="st-key-chart_"] {
  background: var(--glass);
  -webkit-backdrop-filter: blur(28px) saturate(180%); backdrop-filter: blur(28px) saturate(180%);
  border: 1px solid var(--glass-border); border-radius: 22px; box-shadow: var(--shadow);
}
.st-key-assistant { padding: 20px 18px 14px; }
[class*="st-key-chart_"] { padding: 16px 18px 6px; margin-bottom: 22px; }

/* Sidebar: translucent source list */
[data-testid="stSidebar"] {
  background: rgba(246,246,250,.66);
  -webkit-backdrop-filter: blur(30px) saturate(180%); backdrop-filter: blur(30px) saturate(180%);
  border-right: 1px solid var(--separator);
}
/* Works with both radio DOMs: Streamlit <=1.5x (baseweb) and >=1.6x (react-aria) */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[data-testid="stSidebar"] [data-testid="stRadio"], [data-testid="stSidebar"] [data-testid="stRadio"] > div { width: 100% !important; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; width: 100%; display: flex; flex-direction: column; align-items: stretch; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; box-sizing: border-box; padding: 8px 12px; border-radius: 10px; margin: 0;
  transition: background .15s ease, transform 100ms ease-out;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--fill); }
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 15px; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--blue); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: #fff; font-weight: 600; }

/* Text inputs, selects and the chat field: rounded fills, no hard outlines */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stChatInput"] > div {
  border-radius: 12px !important; border-color: transparent !important; background: var(--fill) !important;
}
[data-testid="stChatInput"] > div { border-radius: 20px !important; background: rgba(255,255,255,.9) !important;
  box-shadow: 0 0 0 1px var(--separator); }

/* Tabs as an iOS segmented control */
[role="tablist"] { background: var(--fill); border-radius: 10px; padding: 2px; gap: 0; width: fit-content; max-width: 100%; overflow-x: auto;
                   border: none !important; box-shadow: none !important; }
[role="tablist"]::after, [role="tablist"]::before { display: none !important; }
[data-testid="stTab"] { padding: 6px 18px !important; border-radius: 8px; margin: 0 !important; height: auto !important;
                        transition: background .15s ease; }
[data-testid="stTab"] p { font-size: 13.5px !important; font-weight: 500; color: var(--label) !important; }
[data-testid="stTab"][aria-selected="true"] { background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 1px rgba(0,0,0,.04); }
[data-testid="stTab"][aria-selected="true"] p { font-weight: 600; }
[role="tablist"] .react-aria-SelectionIndicator,
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[role="tabpanel"] { padding-top: 18px; }

/* Header */
.eyebrow { font-size: 13px; font-weight: 600; letter-spacing: .02em; color: var(--secondary); text-transform: uppercase; }
.large-title { font-size: 34px; line-height: 41px; font-weight: 700; letter-spacing: -.022em; margin: 2px 0 4px; }
.subhead { font-size: 15px; color: var(--secondary); }
.header { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 22px; }
.header-side { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.capsule { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px;
           font-size: 13px; font-weight: 600; background: var(--fill); color: var(--label); white-space: nowrap; }
.capsule img.icon { width: 14px; height: 14px; }
.capsule-critical { background: var(--red-fill); color: var(--red); }
.capsule-quiet { font-weight: 500; color: var(--secondary); }

/* Health-style measure tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 26px; }
.tile { padding: 14px 16px 10px; }
.tile-head { display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; white-space: nowrap; }
.tile-head span { overflow: hidden; text-overflow: ellipsis; }
.tile-head img.icon { width: 15px; height: 15px; }
.tile-date { margin-left: auto; font-weight: 400; color: var(--secondary); }
.tile-value { font-size: 30px; font-weight: 600; letter-spacing: -.02em; margin-top: 10px; line-height: 1.1; }
.tile-unit { font-size: 14px; font-weight: 500; color: var(--secondary); margin-left: 3px; letter-spacing: 0; }
.tile-delta { font-size: 12.5px; color: var(--secondary); margin-top: 2px; }
.tile img.spark { width: 100%; height: 38px; display: block; margin-top: 6px; }

/* iOS inset grouped lists */
.group-title { font-size: 12.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .03em;
               color: var(--secondary); margin: 2px 0 7px 16px; }
.group { overflow: hidden; margin-bottom: 22px; }
.cell { display: flex; align-items: center; gap: 12px; padding: 11px 16px; position: relative; }
.cell + .cell::before { content: ""; position: absolute; top: 0; left: 16px; right: 0; height: 1px; background: var(--separator); }
.cell-main { flex: 1; min-width: 0; }
.cell-title { font-size: 15px; font-weight: 500; }
.cell-sub { font-size: 13px; color: var(--secondary); margin-top: 1px; }
.status { font-size: 13px; color: var(--secondary); white-space: nowrap; display: inline-flex; align-items: center; gap: 6px; }
.status i { width: 7px; height: 7px; border-radius: 50%; background: var(--tertiary); display: inline-block; }
.status.active i { background: var(--blue); }
.status.active { color: var(--label); }
.empty { color: var(--secondary); font-size: 14px; padding: 14px 16px; }

/* Assistant: Messages-style bubbles */
.assistant-title { font-size: 20px; font-weight: 700; letter-spacing: -.01em; }
.assistant-sub { font-size: 13px; color: var(--secondary); margin: 2px 0 12px; }
.st-key-messages { background: transparent; border: none !important; }
.st-key-messages [data-testid="stChatMessage"] { background: transparent; padding: 3px 0; gap: 0; }
.st-key-messages [data-testid^="stChatMessageAvatar"] { display: none; }
.st-key-messages [data-testid="stChatMessageContent"] {
  background: var(--fill); border-radius: 18px; padding: 9px 14px 4px; max-width: 92%; flex: 0 1 auto;
  font-size: 14.5px;
}
.st-key-messages [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { justify-content: flex-end; }
.st-key-messages [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: var(--blue); max-width: 82%;
}
.st-key-messages [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * { color: #fff; }
.st-key-messages [data-testid="stExpander"] details { border: none; background: rgba(255,255,255,.55); border-radius: 12px; }
.st-key-messages [data-testid="stExpander"] summary p { font-size: 12.5px; color: var(--secondary); }
.source { padding: 8px 0; border-top: 1px solid var(--separator); }
.source:first-child { border-top: none; }
.source-title { font-size: 12px; font-weight: 600; color: var(--label); }
.source-score { font-weight: 400; color: var(--secondary); }
.source-text { font-size: 12px; color: var(--secondary); white-space: pre-wrap; margin-top: 3px;
               max-height: 120px; overflow: auto; }
.hint { color: var(--secondary); font-size: 14px; text-align: center; padding: 110px 28px 0; text-wrap: balance; }

/* Pills (suggestions, filters) as soft capsules */
[data-testid="stButtonGroup"] button { border-radius: 999px !important; border: none !important;
  background: var(--fill) !important; font-size: 13px; }
[data-testid="stButtonGroup"] button[kind$="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--label) !important; color: #fff !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: #fff !important; }
/* Response: feedback on press, instantly */
[data-testid="stTab"], [data-testid="stButtonGroup"] button {
  transition: transform 100ms ease-out, background .15s ease;
}
[data-testid="stTab"]:active, [data-testid="stSidebar"] [role="radiogroup"] label:active, [data-testid="stButtonGroup"] button:active {
  transform: scale(.97);
}
/* Bright top edge: light catching the material */
.glass, .st-key-assistant, [class*="st-key-chart_"] { box-shadow: inset 0 1px 0 rgba(255,255,255,.9), var(--shadow); }

/* Disclosure groups */
[data-testid="stExpander"] details { border-radius: 14px; border-color: var(--separator); background: rgba(255,255,255,.5); }

/* Laptop widths: the assistant moves below the chart instead of being squeezed */
@media (max-width: 1280px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-assistant) { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"]:has(.st-key-assistant) > [data-testid="stColumn"] { min-width: 100%; }
}

/* Accessibility: honour the user's system settings */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
  [data-testid="stTab"]:active, [data-testid="stSidebar"] [role="radiogroup"] label:active { transform: none; }
}
@media (prefers-reduced-transparency: reduce) {
  .glass, .st-key-assistant, [class*="st-key-chart_"], [data-testid="stSidebar"] {
    background: #fff !important; -webkit-backdrop-filter: none !important; backdrop-filter: none !important;
  }
}
@media (prefers-contrast: more) {
  :root { --secondary: rgba(60,60,67,.85); --separator: rgba(60,60,67,.35); }
  .glass, .st-key-assistant, [class*="st-key-chart_"] { background: #fff; border-color: rgba(60,60,67,.45); }
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data access (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the patient index (the first run embeds the records and can take a few minutes)...")
def load_index():
    rag_pipeline.configure_settings()
    return rag_pipeline.get_index(limit=PATIENT_LIMIT)


@st.cache_data
def indexed_patients(_index):
    """Patients present in the index, read from its metadata so the list always matches it."""
    patients = {n.metadata["patient_id"]: n.metadata["patient_name"] for n in _index.docstore.docs.values()}
    return dict(sorted(patients.items(), key=lambda item: item[1]))


@st.cache_data
def load_patient(patient_id):
    """Synthea file names contain the patient UUID, e.g. Abdul218_Harris789_<uuid>.json."""
    for path in rag_pipeline.list_bundle_files():
        if patient_id in os.path.basename(path):
            bundle = load_fhir_bundle(path)
            series = observation_series(bundle)
            return {
                "header": patient_header(bundle),
                "allergies": allergies(bundle),
                "problems": problem_list(bundle),
                "medications": medication_list(bundle),
                "series": series,
                "measures": key_measures(series),
                "timeline": timeline_events(bundle),
            }
    return None


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------
def esc(text):
    return html.escape(str(text))


def nice_date(iso):
    """'2016-04-10' -> '10 Apr 2016'."""
    try:
        return date.fromisoformat(str(iso)[:10]).strftime("%-d %b %Y")
    except ValueError:
        return str(iso)


def unit(u):
    return UNITS.get(u, u)


def svg_img(svg, css_class, alt=""):
    """st.html sanitises inline <svg>, so SVGs are embedded as data-URI images."""
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="{css_class}" src="data:image/svg+xml;base64,{data}" alt="{esc(alt)}">'


def icon(path, color):
    return svg_img(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
                   f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>', "icon")


WARNING_ICON = "M12 3 2 21h20L12 3zM12 10v5M12 18h.01"


def sparkline(values, color, label):
    """Inline SVG trend line with a soft area fill; native tooltip lists the readings."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    w, h, pad = 200, 38, 3
    pts = [(i * w / (len(values) - 1), pad + (h - 2 * pad) * (1 - (v - lo) / span)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    gid = "g" + "".join(ch for ch in label if ch.isalnum())
    readings = ", ".join(f"{v:g}" for v in values)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0" stop-color="{color}" stop-opacity=".22"/><stop offset="1" stop-color="{color}" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        f'<polygon points="0,{h} {line} {w},{h}" fill="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>'
    )
    title = f"{label}: last {len(values)} readings: {readings}"
    return svg_img(svg, "spark", title).replace("<img ", f'<img title="{esc(title)}" ', 1)


def status_html(status):
    label = {"active": "Active", "resolved": "Resolved", "stopped": "Stopped"}.get(status, status.capitalize())
    return f'<span class="status {"active" if status == "active" else ""}"><i></i>{esc(label)}</span>'


def grouped_list(title, cells, empty="Nothing recorded"):
    """iOS inset grouped list. cells: list of (title, subtitle, accessory_html)."""
    body = "".join(
        f'<div class="cell"><div class="cell-main"><div class="cell-title">{esc(t)}</div>'
        f'<div class="cell-sub">{esc(s)}</div></div>{acc}</div>'
        for t, s, acc in cells
    ) or f'<div class="empty">{esc(empty)}</div>'
    title_html = f'<div class="group-title">{esc(title)}</div>' if title else ""
    return f'{title_html}<div class="glass group">{body}</div>'


def apple_chart(chart, height):
    return chart.properties(height=height, background="transparent").configure(font=FONT).configure_view(
        stroke=None
    ).configure_axis(
        labelColor="#8e8e93", labelFontSize=11, titleColor="#8e8e93", titleFontWeight=500,
        domain=False, ticks=False, gridColor="rgba(60,60,67,0.10)", gridDash=[2, 3],
    ).configure_legend(labelColor="#3c3c43", labelFontSize=12, symbolType="circle", orient="top")


# --------------------------------------------------------------------------
# Header and measure tiles
# --------------------------------------------------------------------------
def render_header(p):
    h = p["header"]
    meta = [h["gender"].capitalize()]
    if h["age"] is not None:
        meta.append(f"{h['age']} years")
    if h["birth_date"]:
        meta.append(f"Born {nice_date(h['birth_date'])}")
    if h["last_encounter"]:
        meta.append(f"Last seen {nice_date(h['last_encounter'])}")
    capsules = "".join(
        f'<span class="capsule {"capsule-critical" if a["criticality"] == "high" else ""}">'
        f'{icon(WARNING_ICON, "#D70015" if a["criticality"] == "high" else "#3c3c43")}Allergy · {esc(a["substance"])}'
        f'{" · high risk" if a["criticality"] == "high" else ""}</span>'
        for a in p["allergies"]
    ) or '<span class="capsule capsule-quiet">No known allergies</span>'
    if h["deceased"]:
        capsules = f'<span class="capsule">Deceased {esc(nice_date(h["deceased"]))}</span>' + capsules
    st.html(f"""
<div class="header">
  <div><div class="eyebrow">Patient · ID {esc(h['id'][:8])}</div>
       <div class="large-title">{esc(h['name'])}</div>
       <div class="subhead">{esc(' · '.join(meta))}</div></div>
  <div class="header-side">{capsules}</div>
</div>""")


def render_tiles(measures):
    tiles = []
    for m in measures:
        color, path = TILE_STYLE[m["label"]]
        if m["label"] == "Blood pressure" and "diastolic" in m:
            value = f"{m['value']:.0f}/{m['diastolic']:.0f}"
        else:
            value = f"{m['value']:.1f}".rstrip("0").rstrip(".")
        if m["delta"] is None:
            delta = "First reading"
        elif abs(m["delta"]) < 0.05:
            delta = "Unchanged since previous"
        else:  # direction shown with an arrow; neutral colour because "up" is not good or bad for every measure
            delta = f"{'↑' if m['delta'] > 0 else '↓'} {abs(m['delta']):.1f} since previous"
        tiles.append(f"""
<div class="glass tile">
  <div class="tile-head">{icon(path, color)}<span>{esc(m['label'])}</span><span class="tile-date" title="Latest reading {esc(nice_date(m['date']))}">{esc(date.fromisoformat(m['date']).strftime('%b %Y'))}</span></div>
  <div class="tile-value">{esc(value)}<span class="tile-unit">{esc(unit(m['unit']))}</span></div>
  <div class="tile-delta">{esc(delta)}</div>
  {sparkline(m['history'], color, m['label'])}
</div>""")
    if tiles:
        st.html(f'<div class="tiles">{"".join(tiles)}</div>')


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
def render_summary(p):
    problems = [x for x in p["problems"] if x["status"] == "active" and not x["social"]]
    meds = [m for m in p["medications"] if m["status"] == "active"]
    left, right = st.columns(2, gap="medium")
    left.html(grouped_list("Active problems",
                           [(x["problem"], f"Since {nice_date(x['onset'])}", "") for x in problems],
                           "No active problems"))
    right.html(grouped_list("Active medication",
                            [(m["medication"], f"Since {nice_date(m['first'])}", "") for m in meds],
                            "No active medication"))


def render_problems(p):
    view = st.segmented_control("View", ["Clinical", "Social history"], default="Clinical",
                                label_visibility="collapsed", key="problem_view") or "Clinical"
    rows = [x for x in p["problems"] if x["social"] == (view == "Social history")]
    for status, title in (("active", "Active"), ("resolved", "Resolved")):
        cells = []
        for x in rows:
            if (x["status"] == "active") != (status == "active"):
                continue
            sub = f"Onset {nice_date(x['onset'])}"
            if x["resolved"]:
                sub += f" · resolved {nice_date(x['resolved'])}"
            cells.append((x["problem"], sub, status_html(x["status"])))
        st.html(grouped_list(title, cells, f"No {title.lower()} entries"))


def render_medications(p):
    for active, title in ((True, "Active"), (False, "Previous")):
        cells = [
            (m["medication"],
             f"{nice_date(m['first'])} – {nice_date(m['last'])} · {m['prescriptions']} "
             f"prescription{'s' if m['prescriptions'] != 1 else ''}",
             status_html(m["status"]))
            for m in p["medications"] if (m["status"] == "active") == active
        ]
        st.html(grouped_list(title, cells, f"No {title.lower()} medication"))


def trend_chart(df, unit_label, multi):
    domain = ["Systolic", "Diastolic"] if multi else [df["series"].iloc[0]]
    color = alt.Color("series:N", scale=alt.Scale(domain=domain, range=[BLUE, ORANGE]),
                      legend=alt.Legend(title=None) if multi else None)
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(grid=False, format="%Y", labelPadding=8)),
        y=alt.Y("value:Q", title=unit_label, scale=alt.Scale(zero=False, nice=True), axis=alt.Axis(tickCount=5)),
    )
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    tooltip = [alt.Tooltip("date:T", title="Date", format="%-d %b %Y"),
               alt.Tooltip("series:N", title="Measure"), alt.Tooltip("value:Q", title=unit_label, format=".1f")]
    layers = [
        base.mark_line(strokeWidth=2, interpolate="monotone").encode(color=color),
        base.mark_point(filled=True, size=70, stroke="white", strokeWidth=2, opacity=1).encode(
            color=color, tooltip=tooltip).add_params(hover),
        base.mark_rule(color="#8e8e93", strokeDash=[3, 3]).encode(
            opacity=alt.condition(hover, alt.value(0.7), alt.value(0))).transform_filter(hover),
    ]
    if multi:  # direct labels so identity is never colour-only
        last = df.sort_values("date").groupby("series").tail(1)
        layers.append(alt.Chart(last).mark_text(align="left", dx=9, fontSize=12, fontWeight=500, color="#3c3c43")
                      .encode(x="date:T", y="value:Q", text="series:N"))
    return apple_chart(alt.layer(*layers), 300)


def render_trends(p):
    series = p["series"]
    if not series:
        st.html(grouped_list("", [], "No numeric observations recorded"))
        return
    options = (["Blood pressure"] if "Systolic Blood Pressure" in series else []) + sorted(
        (n for n in series if "Blood Pressure" not in n), key=lambda n: -len(series[n]))
    choice = st.selectbox("Measure", options, key="trend_measure", label_visibility="collapsed")
    names = [n for n in ("Systolic Blood Pressure", "Diastolic Blood Pressure") if n in series] \
        if choice == "Blood pressure" else [choice]
    df = pd.DataFrame([{**pt, "series": n.replace(" Blood Pressure", "")} for n in names for pt in series[n]])
    df["date"] = pd.to_datetime(df["date"])
    u = unit(df["unit"].iloc[0])
    with st.container(key="chart_trend"):
        st.altair_chart(trend_chart(df, u, multi=len(names) > 1), width="stretch", theme=None)
    with st.expander("Show all readings"):
        rows = df.pivot_table(index="date", columns="series", values="value").sort_index(ascending=False)
        cells = [(nice_date(d.date().isoformat()), "",
                  f'<span class="status active">{esc(" / ".join(f"{v:g}" for v in r.dropna().values))} {esc(u)}</span>')
                 for d, r in rows.iterrows()]
        st.html(grouped_list("", cells))


TIMELINE_FILTERS = {"Key events": ["Diagnosis", "Medication"], "Encounters": ["Encounter"],
                    "Everything": ["Diagnosis", "Medication", "Encounter"]}


def render_timeline(p):
    events = pd.DataFrame(p["timeline"])
    if events.empty:
        st.html(grouped_list("", [], "No events recorded"))
        return
    choice = st.segmented_control("Show", list(TIMELINE_FILTERS), default="Key events",
                                  label_visibility="collapsed", key="timeline_filter") or "Key events"
    events = events[events["category"].isin(TIMELINE_FILTERS[choice])].copy()
    if events.empty:
        st.html(grouped_list("", [], "Nothing to show for this filter"))
        return
    events["when"] = pd.to_datetime(events["date"])
    order = ["Diagnosis", "Medication", "Encounter"]
    chart = alt.Chart(events).mark_circle(size=110, stroke="white", strokeWidth=2, opacity=1).encode(
        x=alt.X("when:T", title=None, axis=alt.Axis(grid=False, format="%Y")),
        y=alt.Y("category:N", title=None, sort=order,
                axis=alt.Axis(labelFontSize=12, labelColor="#3c3c43", labelLimit=200, minExtent=90, labelPadding=10)),
        color=alt.Color("category:N", scale=alt.Scale(domain=order, range=[BLUE, ORANGE, TEAL]), legend=None),
        tooltip=[alt.Tooltip("when:T", title="Date", format="%-d %b %Y"),
                 alt.Tooltip("category:N", title="Type"), alt.Tooltip("event:N", title="Event")],
    )
    with st.container(key="chart_timeline"):
        st.altair_chart(apple_chart(chart, 64 * events["category"].nunique() + 30), width="stretch", theme=None)
    # Grouped by year, newest first, like a Health or Photos timeline
    for year, group in sorted(events.groupby(events["when"].dt.year), key=lambda g: -g[0]):
        cells = [(r["event"], f"{nice_date(r['date'])} · {r['category']}", "")
                 for _, r in group.sort_values("when", ascending=False).iterrows()]
        st.html(grouped_list(str(year), cells))


# --------------------------------------------------------------------------
# Assistant
# --------------------------------------------------------------------------
def queue_suggestion():
    st.session_state.pending_question = st.session_state.suggestion
    st.session_state.suggestion = None


def render_sources(sources):
    with st.expander(f"Sources · {len(sources)} excerpts from the record"):
        st.html("".join(
            f'<div class="source"><div class="source-title">{esc(s["section"])} '
            f'<span class="source-score">· similarity {s["score"]:.2f}</span></div>'
            f'<div class="source-text">{esc(s["text"])}</div></div>'
            for s in sources
        ))


def render_assistant(index, patient_id, first_name):
    history = st.session_state.setdefault("history", {}).setdefault(patient_id, [])
    with st.container(key="assistant"):
        st.html('<div class="assistant-title">Ask about this record</div>'
                f'<div class="assistant-sub">Answers use only {esc(first_name)}\'s record. '
                'The excerpts behind each answer are one tap away.</div>')
        messages = st.container(height=380, key="messages")
        if not history:
            st.pills("Suggestions", SUGGESTED_QUESTIONS, key="suggestion",
                     on_change=queue_suggestion, label_visibility="collapsed")
        asked = st.session_state.get("questions_asked", 0)
        limit_reached = asked >= MAX_QUESTIONS
        typed = st.chat_input("Demo limit reached for this session" if limit_reached else "Ask a question",
                              disabled=limit_reached)
    question = typed or st.session_state.pop("pending_question", None)
    if question and limit_reached:
        question = None
    if question:
        st.session_state.questions_asked = asked + 1

    with messages:
        if not history and not question:
            st.html('<div class="hint">Ask a question in your own words, or start with a suggestion below.</div>')
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources(message["sources"])
        if not question:
            return
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Reading the record…"):
                response = rag_pipeline.query_patient(index, patient_id, question)
            answer = str(response)
            sources = [{"section": n.metadata.get("section", ""), "score": n.score or 0.0, "text": n.get_content()}
                       for n in response.source_nodes]
            st.markdown(answer)
            render_sources(sources)
        history.append({"role": "assistant", "content": answer, "sources": sources})
    if len(history) == 2:
        st.rerun()  # hide the suggestions once the conversation has started


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
index = load_index()
patients = indexed_patients(index)

with st.sidebar:
    st.html('<div class="large-title" style="font-size:28px">Patients</div>'
            f'<div class="subhead" style="margin-bottom:10px">{len(patients)} records</div>')
    query = st.text_input("Search", placeholder="Search", label_visibility="collapsed").strip().lower()
    matches = [pid for pid, name in patients.items() if query in name.lower()] or list(patients)
    patient_id = st.radio("Patient", matches, format_func=patients.get, label_visibility="collapsed")
    st.html('<div class="subhead" style="font-size:12px;margin-top:18px">Synthetic FHIR R4 records (Synthea). '
            'A portfolio project, not a medical device.</div>')

patient = load_patient(patient_id)
if patient is None:
    st.error("This patient's FHIR file was not found in data/fhir.")
    st.stop()

render_header(patient)
render_tiles(patient["measures"])

chart_col, assistant_col = st.columns([1.85, 1], gap="large")
with chart_col:
    summary, problems, medications, trends, timeline = st.tabs(
        ["Summary", "Problems", "Medications", "Trends", "Timeline"])
    with summary:
        render_summary(patient)
    with problems:
        render_problems(patient)
    with medications:
        render_medications(patient)
    with trends:
        render_trends(patient)
    with timeline:
        render_timeline(patient)
with assistant_col:
    render_assistant(index, patient_id, patient["header"]["name"].split()[0])
