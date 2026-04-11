from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

TOP_K = 4  # Number of chunks to retrieve


def load_vectorstore(path: str) -> FAISS:
    """Load a saved FAISS vectorstore from disk."""
    from rag.embedder import get_embeddings
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def search_docs(vectorstore: FAISS, query: str, k: int = TOP_K) -> List[Document]:
    """
    Semantic similarity search on the vectorstore.
    Returns top-k most relevant document chunks.
    """
    if vectorstore is None:
        raise ValueError("Vectorstore is not initialized. Please process documents first.")

    results = vectorstore.similarity_search(query, k=k)
    return results