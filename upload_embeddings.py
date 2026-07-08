from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase_client import supabase
from pdf2image import convert_from_path
import pytesseract
import os

model = SentenceTransformer("BAAI/bge-m3")


def extract_text(file_path):
    """
    Extract raw text from a docx, pdf, or txt file.
    PDFs are processed using OCR (Tesseract).
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".pdf":
        images = convert_from_path(file_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
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

    rows = []

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        rows.append({
            "content": chunk,
            "embedding": embedding,
            "file_name": file_name
        })

    batch_size = 20

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        for attempt in range(3):
            try:
                supabase.table("documents").insert(batch).execute()
                break
            except Exception as e:
                print(f"Batch insert attempt {attempt+1} failed: {e}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Failed to insert batch starting at row {i} after 3 attempts")

    return len(chunks)