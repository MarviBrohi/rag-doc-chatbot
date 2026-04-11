from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Tuple
import os

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_embeddings():
    """Load embedding model - tries new package first, falls back to old."""
    try:
        # Try new recommended package first
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fall back to old import if new package not installed
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def embed_documents(texts: List[str], metadatas: List[dict]) -> Tuple[FAISS, int]:
    """Split texts into chunks and embed into FAISS vectorstore."""

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No chunks created. Check your PDF content.")

    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore, len(chunks)
    except Exception as e:
        error_msg = str(e)
        if "huggingface.co" in error_msg or "resolve" in error_msg or "connect" in error_msg:
            raise ConnectionError(
                "Cannot connect to HuggingFace to download the embedding model.\n\n"
                "FIXES:\n"
                "1. Check your internet connection\n"
                "2. Run this command: pip install langchain-huggingface\n"
                "3. If behind a firewall, try using a VPN\n"
                "4. The model only needs to download ONCE — after that it works offline"
            )
        raise e
