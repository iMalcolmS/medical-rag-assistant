"""Pipeline tests with mock models: no API key, no model download, no network."""

import pytest
from llama_index.core import MockEmbedding, Settings
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter

from src.rag_pipeline import get_index, list_bundle_files, load_documents, query_patient


@pytest.fixture(autouse=True)
def mock_models():
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)


def test_non_json_files_are_skipped(fhir_dir):
    files = list_bundle_files(str(fhir_dir))
    assert len(files) == 2
    assert all(f.endswith(".json") for f in files)


def test_patient_id_is_kept_out_of_the_embedded_text(fhir_dir):
    doc = load_documents(str(fhir_dir))[0]
    embedded = doc.get_content(metadata_mode="embed")
    assert "patient_id:" not in embedded
    assert "patient_name: Abdul Harris" in embedded


@pytest.mark.parametrize("patient_id", ["patient-a", "patient-b"])
def test_retrieval_never_mixes_patients(fhir_dir, tmp_path, patient_id):
    index = get_index(str(fhir_dir), str(tmp_path / "storage"))
    response = query_patient(index, patient_id, "Which conditions does the patient have?")
    assert response.source_nodes
    assert {n.metadata["patient_id"] for n in response.source_nodes} == {patient_id}


def test_index_is_persisted_and_reloaded(fhir_dir, tmp_path):
    storage = tmp_path / "storage"
    get_index(str(fhir_dir), str(storage))
    assert storage.is_dir()
    reloaded = get_index(str(fhir_dir), str(storage))
    response = query_patient(reloaded, "patient-b", "medications?")
    assert {n.metadata["patient_id"] for n in response.source_nodes} == {"patient-b"}
