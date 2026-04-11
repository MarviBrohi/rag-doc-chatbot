import fitz  # PyMuPDF
import io
from typing import List, Tuple


def extract_text_from_pdfs(uploaded_files) -> Tuple[List[str], List[dict]]:
    """
    Extract text and metadata from uploaded PDF files.
    Returns: (list of text chunks by page, list of metadata dicts)
    """
    all_texts = []
    all_metadatas = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        file_name = uploaded_file.name

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if text:  # skip empty pages
                all_texts.append(text)
                all_metadatas.append({
                    "source": file_name,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                })

        doc.close()

    if not all_texts:
        raise ValueError("No readable text found in the uploaded PDFs.")

    return all_texts, all_metadatas