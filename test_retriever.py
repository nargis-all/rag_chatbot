from retriever import retrieve

question = "What documents are required for temporary import?"

results = retrieve(question)

for item in results:
    print("=" * 80)
    print(item["similarity"])
    print(item["content"])