from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

doc = Document("data/lexuz_5535133.docx")

text = ""

for p in doc.paragraphs:
    text += p.text + "\n"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_text(text)

print("Chunks:", len(chunks))