"""
Document ingestion and retriever construction.
"""
from __future__ import annotations

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


def load_documents(settings: Settings):
    """Load documents from the configured Wikipedia sources."""

    documents = []
    for url in settings.source_urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    return documents


def split_documents(docs, settings: Settings):
    """Split documents into overlapping chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_retriever(settings: Settings):
    """Construct a FAISS-backed retriever from web documents."""

    docs = load_documents(settings)
    chunks = split_documents(docs, settings)
    embeddings = HuggingFaceEmbeddings(model_name=settings.huggingface_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()
