import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(context, question):
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If the answer is not in the context, reply exactly:
"I couldn't find the answer in the provided document."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content