from retriever import retrieve
from llm import ask_llm

question = input("Ask a question: ")

results = retrieve(question, top_k=2)

context = "\n\n".join([doc["content"][:800] for doc in results])

answer = ask_llm(context, question)

print("\n===== ANSWER =====\n")
print(answer)