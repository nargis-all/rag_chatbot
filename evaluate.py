import pandas as pd
from retriever import retrieve
from llm import ask_llm

df = pd.read_excel("questions.xlsx")

predicted_answers = []

for question in df["Question"]:
    print(f"Processing: {question}")

    results = retrieve(question, top_k=2)

    context = "\n\n".join([doc["content"][:800] for doc in results])

    answer = ask_llm(context, question)

    predicted_answers.append(answer)

df["Predicted Answer"] = predicted_answers

df.to_excel("results.xlsx", index=False)

print("Done!")