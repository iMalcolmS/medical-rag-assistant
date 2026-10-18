"""Streamlit front end: pick a patient, ask questions about their record, browse the timeline.

Run with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from src import rag_pipeline
from src.fhir_loader import (
    extract_medications,
    extract_resources,
    extract_timeline,
    load_fhir_bundle,
)

# Number of patients indexed on the first run. Changing it requires deleting
# the storage/ folder so the index is rebuilt with the new set of patients.
PATIENT_LIMIT = int(os.environ.get("PATIENT_LIMIT", 20))

EXAMPLE_QUESTIONS = [
    "Which chronic conditions does the patient have?",
    "What medications is the patient currently taking?",
    "How has the patient's blood pressure changed over time?",
]

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the patient index (first run embeds the records, this can take a few minutes)...")
def load_index():
    rag_pipeline.configure_settings()
    return rag_pipeline.get_index(limit=PATIENT_LIMIT)


@st.cache_data
def indexed_patients(_index):
    """Patients present in the index, read from its metadata so the list always matches it."""
    patients = {}
    for node in _index.docstore.docs.values():
        patients[node.metadata["patient_id"]] = node.metadata["patient_name"]
    return dict(sorted(patients.items(), key=lambda item: item[1]))


@st.cache_data
def load_patient_bundle(patient_id):
    """Synthea file names contain the patient UUID, e.g. Abdul218_Harris789_<uuid>.json."""
    for path in rag_pipeline.list_bundle_files():
        if patient_id in os.path.basename(path):
            return load_fhir_bundle(path)
    return None


# --------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------
def render_overview(bundle):
    conditions = [
        c for c in extract_resources(bundle, "Condition")
        if c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active"
    ]
    medications = [m for m in extract_medications(bundle) if m["status"] == "active"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active conditions")
        if conditions:
            rows = []
            for c in conditions:
                name = c.get("code", {}).get("text") or c.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
                rows.append({
                    "Condition": name,
                    "Since": c.get("onsetDateTime", "")[:10],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No active conditions recorded.")
    with col2:
        st.subheader("Active medications")
        if medications:
            unique = pd.DataFrame(medications).drop_duplicates("name").drop(columns="status")
            st.dataframe(unique.rename(columns={"name": "Medication", "date": "Prescribed"}),
                         hide_index=True, width="stretch")
        else:
            st.info("No active medications recorded.")


def render_timeline(bundle):
    events = pd.DataFrame(extract_timeline(bundle))
    if events.empty:
        st.info("No events recorded.")
        return
    events["date"] = events["date"].str[:10]
    types = st.multiselect("Event types", sorted(events["type"].unique()),
                           default=["Condition", "Medication"])
    filtered = events[events["type"].isin(types)].sort_values("date", ascending=False)
    st.caption(f"{len(filtered)} of {len(events)} events")
    st.dataframe(filtered.rename(columns=str.capitalize), hide_index=True, width="stretch")


def render_chat(index, patient_id):
    history = st.session_state.setdefault("history", {}).setdefault(patient_id, [])

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    st.caption("Try: " + " · ".join(f"*{q}*" for q in EXAMPLE_QUESTIONS))
    question = st.chat_input("Ask a question about this patient's record")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the record..."):
            response = rag_pipeline.query_patient(index, patient_id, question)
        answer = str(response)
        sources = [
            {"section": n.metadata.get("section", ""), "score": n.score or 0.0, "text": n.get_content()}
            for n in response.source_nodes
        ]
        st.markdown(answer)
        render_sources(sources)
    history.append({"role": "assistant", "content": answer, "sources": sources})


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)} record excerpts)"):
        for s in sources:
            st.markdown(f"**{s['section']}** · similarity {s['score']:.2f}")
            st.code(s["text"], language=None)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.title("🩺 Medical RAG Assistant")

index = load_index()
patients = indexed_patients(index)

with st.sidebar:
    st.header("Patient")
    patient_id = st.selectbox("Select a patient", list(patients), format_func=patients.get)
    st.caption(f"ID: {patient_id}")
    st.divider()
    st.caption(
        f"{len(patients)} patients indexed. All data is synthetic (Synthea, FHIR R4). "
        "This is a portfolio project, not a medical device: do not use it for clinical decisions."
    )

bundle = load_patient_bundle(patient_id)
chat_tab, overview_tab, timeline_tab = st.tabs(["💬 Ask", "📋 Overview", "🗓️ Timeline"])

with chat_tab:
    render_chat(index, patient_id)
with overview_tab:
    if bundle:
        render_overview(bundle)
    else:
        st.warning("Patient file not found in data/fhir.")
with timeline_tab:
    if bundle:
        render_timeline(bundle)
    else:
        st.warning("Patient file not found in data/fhir.")
