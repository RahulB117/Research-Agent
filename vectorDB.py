import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma # For using Chroma as a vector store
from langchain.embeddings import OpenAIEmbeddings # For using OpenAI's embeddings
from langchain.schema import Document # For storing documents in the vector store
from typing import Dict # Store metadata as a dictionary

load_dotenv()
PERSIST_DIR = "chroma_db"
EMBEDDINGS = OpenAIEmbeddings()


def initialize_vector_store() -> Chroma:
    """ Chroma automatically creates the directory if it doesn't exist. """
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDINGS)

def validate_entry(summary: str, query: str) -> bool:
    """Filter out junk, empty, or repeated summaries."""
    if not summary or len(summary.strip()) < 100:
        return False

    failure_phrases = ["Agent stopped", "I'm sorry", "None"]
    if any(phrase in summary for phrase in failure_phrases):
        return False

    summary_lower = summary.strip().lower()
    query_lower = query.strip().lower()

    if summary_lower == query_lower:
        return False

    return True

def add_to_vector_store(topic: str, summary: str, metadata: Dict, query: str):
    """Store a valid research entry in the vector store."""
    print("🔍 Summary Preview:", summary[:100])
    print("📁 VectorDB Path Exists:", os.path.exists(PERSIST_DIR))
    if not validate_entry(summary, query):
        print("🛑 Skipping junk or echo response.")
        return

    vectorstore = initialize_vector_store()

    cleaned_metadata = {
        k: (", ".join(v) if isinstance(v, list) else v) for k, v in metadata.items()
    }
    doc = Document(page_content=summary, metadata={"topic": topic, **cleaned_metadata})
    vectorstore.add_documents([doc])
    vectorstore.persist()
    print("✅ Research data added to ChromaDB.")