from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Tuple


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_embeddings():
    """Load embedding model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
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

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)
