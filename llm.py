import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(context, question, chat_history=None):
    """
    chat_history: list of (question, answer) tuples from previous turns,
    most recent last. Optional — defaults to no history.
    """

    history_text = ""

    if chat_history:
        for past_q, past_a in chat_history[-3:]:
            history_text += f"Previous Question: {past_q}\nPrevious Answer: {past_a}\n\n"

    prompt = f"""
You are a knowledgeable legal assistant specializing in customs regulations.

Answer ONLY using the context below — do not use outside knowledge.
If the answer is not in the context, reply exactly:
"I couldn't find the answer in the provided document."

When answering:
- Give a DETAILED, well-explained answer, not just a one-line response.
- Explain the relevant rule, article, or concept clearly, as if teaching someone unfamiliar with it.
- If the context includes specific numbers, conditions, exceptions, or procedures, include them explicitly.
- Where relevant, mention which article or section the information comes from.
- Structure longer answers with short paragraphs or bullet points for clarity.
- Do not just restate the question — explain the reasoning and implications where appropriate.

Use the conversation history only to understand follow-up questions
(e.g. "what about X" referring to something discussed earlier).
Do not use the conversation history as a source of facts — only the context below.

{history_text}Context:
{context}

Question:
{question}

Detailed Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content