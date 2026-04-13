import os
from groq import Groq
from langchain.schema import Document
from typing import List


def build_context(docs: List[Document]) -> str:
    """Build formatted context string from retrieved document chunks."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "document")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i} — {source}, Page {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(context_parts)


def get_answer(question: str, docs: List[Document]) -> str:
    """Send question + context to Groq LLaMA3 and return answer."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )

    client = Groq(api_key=api_key)
    context = build_context(docs)

    system_prompt = """You are DocuMind AI, an expert document analyst and assistant.

Your role:
- Answer questions ONLY based on the provided document context
- Be accurate, concise, and helpful
- If the answer is not in the context, clearly say: "This information is not available in the provided documents."
- Structure your answers clearly using markdown when helpful (bullet points, bold text, etc.)
- Always mention which source or page your answer comes from when relevant
- Be professional yet conversational in tone"""

    user_prompt = f"""Based on the following document excerpts, please answer the question.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based only on the context above."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # ✅ Active model on Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content
