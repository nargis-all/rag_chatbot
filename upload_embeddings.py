from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase_client import supabase
import os

model = SentenceTransformer("BAAI/bge-m3")


def extract_text(file_path):
    """
    Extract raw text from a docx, pdf, or txt file.
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def upload_document(file_path, file_name):
    """
    Upload a DOCX, PDF, or TXT file, split it into chunks,
    create embeddings and store them in Supabase.
    """

    text = extract_text(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    for chunk in chunks:

        embedding = model.encode(chunk).tolist()

        supabase.table("documents").insert({
            "content": chunk,
            "embedding": embedding,
            "file_name": file_name
        }).execute()

    return len(chunks)