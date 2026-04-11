from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Tuple


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_embeddings():
    """Load HuggingFace embedding model (cached after first load)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def embed_documents(texts: List[str], metadatas: List[dict]) -> Tuple[FAISS, int]:
    """
    Split texts into chunks and embed them into a FAISS vectorstore.
    Returns: (vectorstore, total chunk count)
    """
    # Build LangChain Document objects
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Document splitting produced no chunks. Check your PDF content.")

    # Embed and store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)