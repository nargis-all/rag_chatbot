from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("BAAI/bge-m3")

embedding = model.encode("Hello world")

print("Embedding length:", len(embedding))