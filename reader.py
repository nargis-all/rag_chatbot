from pathlib import Path
from docx import Document
from pypdf import PdfReader


def read_document(file_path):
    ext = Path(file_path).suffix.lower()

    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".pdf":
        pdf = PdfReader(file_path)
        text = ""

        for page in pdf.pages:
            text += page.extract_text() or ""

        return text

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")