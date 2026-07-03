from sentence_transformers import SentenceTransformer
from supabase_client import supabase
import time

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-m3")

def retrieve(query, top_k=5):
    query_embedding = model.encode(query).tolist()

    for attempt in range(3):
        try:
            response = supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k
                }
            ).execute()

            return response.data

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return []


import time
import httpx

def retrieve_with_retry(question, top_k=2, retries=3, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return retrieve(question, top_k=top_k)
        except httpx.RemoteProtocolError as e:
            print(f"[{attempt}/{retries}] Ulanish uzildi, qayta urinilmoqda... ({e})")
            time.sleep(delay)
        except Exception as e:
            print(f"Kutilmagan xatolik: {e}")
            time.sleep(delay)
    raise RuntimeError(f"{retries} marta urinishdan so'ng ham muvaffaqiyatsiz tugadi")