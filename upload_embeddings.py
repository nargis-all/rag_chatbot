from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase_client import supabase
import os

# Modelni faqat bir marta yuklaymiz
model = SentenceTransformer("BAAI/bge-m3")


def upload_document(file_path, file_name):
    """
    Upload a DOCX file, split it into chunks,
    create embeddings and store them in Supabase.
    """

    doc = Document(file_path)

    text = ""

    for p in doc.paragraphs:
        text += p.text + "\n"

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