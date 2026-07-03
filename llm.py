import ollama

def ask_llm(context, question):
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If the answer is not in the context, reply:
"I couldn't find the answer in the provided document."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]