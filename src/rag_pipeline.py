"""Retrieval-augmented question answering over Synthea FHIR patient records.

Design decisions
----------------
* Patient-scoped retrieval: every query is filtered by ``patient_id`` metadata,
  so the context the LLM sees only ever comes from the selected patient.
* Small chunks (256 tokens): ``all-MiniLM-L6-v2`` only embeds the first 256
  word pieces of a text, so longer chunks would be silently truncated.
* Grounded prompt: the model must answer from the retrieved records only and
  say so when the information is not there, instead of guessing.
* Persisted index: embeddings are computed once and saved to ``storage/``,
  so restarting the app does not re-embed every record.
"""

import os

from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from src.fhir_loader import bundle_to_documents

DATA_DIR = "data/fhir"
STORAGE_DIR = "storage"
# Groq retires models regularly: override with GROQ_MODEL in .env
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
LLM_CONTEXT_WINDOW = 131072
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
TOP_K = 8

QA_PROMPT = PromptTemplate(
    "You are a clinical assistant answering questions about ONE patient's "
    "medical record.\n"
    "Use ONLY the record excerpts below. Do not use outside knowledge about "
    "the patient and do not guess.\n"
    "If the excerpts do not contain the answer, reply exactly: "
    "\"I could not find this information in the patient's record.\"\n"
    "When you mention a condition, medication or measurement, include its date.\n\n"
    "Record excerpts:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


def configure_settings():
    """Set the LLM, embedding model and chunking used by LlamaIndex."""
    # Imported here so the rest of the module (and its tests) work without
    # downloading the embedding model or needing an API key.
    from dotenv import load_dotenv
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.groq import Groq

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    model = os.environ.get("GROQ_MODEL", DEFAULT_LLM_MODEL)
    Settings.llm = Groq(
        model=model,
        api_key=api_key,
        temperature=0,
        context_window=LLM_CONTEXT_WINDOW,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, paragraph_separator="\n"
    )


def list_bundle_files(data_dir=DATA_DIR, limit=None):
    """Sorted FHIR bundle paths. Skips non-JSON files such as macOS .DS_Store."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if limit is not None:
        files = files[:limit]
    return [os.path.join(data_dir, f) for f in files]


def load_documents(data_dir=DATA_DIR, limit=None):
    """Convert FHIR bundles into LlamaIndex Documents with patient metadata."""
    documents = []
    for path in list_bundle_files(data_dir, limit):
        for doc in bundle_to_documents(path):
            documents.append(
                Document(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    # The UUID adds nothing to the meaning of a chunk: keep it
                    # for filtering, but out of the embedding and the prompt.
                    excluded_embed_metadata_keys=["patient_id"],
                    excluded_llm_metadata_keys=["patient_id"],
                )
            )
    return documents


def build_index(documents, storage_dir=STORAGE_DIR):
    """Embed the documents and save the index to disk."""
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=storage_dir)
    return index


def get_index(data_dir=DATA_DIR, storage_dir=STORAGE_DIR, limit=None, rebuild=False):
    """Load the saved index, or build it from the FHIR files if there is none.

    Use ``rebuild=True`` after changing the loader or the number of patients.
    """
    if os.path.isdir(storage_dir) and not rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
    return build_index(load_documents(data_dir, limit), storage_dir)


def query_patient(index, patient_id, question, top_k=TOP_K):
    """Answer a question using only the records of one patient.

    Returns the LlamaIndex Response: ``str(response)`` is the answer and
    ``response.source_nodes`` are the excerpts it was based on.
    """
    filters = MetadataFilters(filters=[ExactMatchFilter(key="patient_id", value=patient_id)])
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        filters=filters,
        text_qa_template=QA_PROMPT,
    )
    return query_engine.query(question)


if __name__ == "__main__":
    # Quick end-to-end check on a small subset: python -m src.rag_pipeline
    configure_settings()
    index = get_index(limit=20)

    first_patient = bundle_to_documents(list_bundle_files(limit=1)[0])[0]["metadata"]
    print(f"\nPatient: {first_patient['patient_name']} ({first_patient['patient_id']})")

    for question in [
        "Which chronic conditions does the patient have?",
        "What medications has the patient been prescribed for heart disease?",
        "What is the patient's blood type?",  # not in Synthea: should say not found
    ]:
        response = query_patient(index, first_patient["patient_id"], question)
        print(f"\nQ: {question}\nA: {response}")
        sections = sorted({n.metadata['section'] for n in response.source_nodes})
        patients = {n.metadata['patient_id'] for n in response.source_nodes}
        print(f"   sources: {sections} | patients in context: {len(patients)}")
