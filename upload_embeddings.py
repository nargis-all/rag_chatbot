from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from supabase_client import supabase

print("Loading model...")

model = SentenceTransformer("BAAI/bge-m3")

print("Reading document...")

doc = Document("data/lexuz_5535133.docx")

text = ""

for p in doc.paragraphs:
    text += p.text + "\n"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):

    embedding = model.encode(chunk).tolist()

    supabase.table("documents").insert({
        "content": chunk,
        "embedding": embedding
    }).execute()

    if (i + 1) % 20 == 0:
        print(f"Uploaded {i+1}/{len(chunks)}")

print("Done!")