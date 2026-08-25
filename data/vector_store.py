"""
Vector store module - creates FAISS index from documents with metadata.
Uses chunk-level indexing with source authority preserved.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from data.loader import load_all_documents


def build_vector_store(api_key: str) -> FAISS:
    """Build FAISS vector store from all documents with metadata."""
    raw_docs = load_all_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    langchain_docs = []
    for doc in raw_docs:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "status": doc["status"],
                "doc_type": doc["doc_type"],
                "authority": doc["authority"],
                "chunk_index": i,
            }
            if "account_id" in doc:
                metadata["account_id"] = doc["account_id"]
            if doc["status"] == "DEPRECATED":
                metadata["warning"] = "DEPRECATED - Do not use for current decisions"

            langchain_docs.append(Document(page_content=chunk, metadata=metadata))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    vector_store = FAISS.from_documents(langchain_docs, embeddings)
    return vector_store
