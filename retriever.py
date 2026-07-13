from sentence_transformers import SentenceTransformer
from supabase_client import supabase
import time
import re

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-m3")


def retrieve(query, file_names=None, top_k=5):
    """
    Retrieve top-k relevant chunks. If file_names is provided (a list),
    only search within those documents. If None, search all documents.
    """

    query_embedding = model.encode(query).tolist()

    for attempt in range(3):

        try:

            response = supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                    "file_names": file_names
                }
            ).execute()

            return response.data

        except Exception as e:

            print(e)
            time.sleep(2)

    return []


def count_word_occurrences(file_name, word):
    """
    Count exact occurrences of a word across the ENTIRE document,
    not just top-k retrieved chunks.
    """

    response = supabase.table("documents") \
        .select("content") \
        .eq("file_name", file_name) \
        .execute()

    full_text = " ".join(row["content"] for row in response.data)

    matches = re.findall(rf"\b{re.escape(word)}\b", full_text, re.IGNORECASE)

    return len(matches)