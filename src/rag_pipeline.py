from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
from dotenv import load_dotenv

load_dotenv()

def build_index(documents):
    docs = [Document(text=doc) for doc in documents]

    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = llm 
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(docs)
    return index 

def query_index(index, question):
    engine = index.as_query_engine()
    response = engine.query(question)
    return str(response)