import streamlit as st
from retriever import retrieve
from llm import ask_llm

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 Customs RAG Chatbot")

question = st.text_input("Ask your question")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching..."):
            results = retrieve(question, top_k=2)

            context = "\n\n".join([doc["content"][:800] for doc in results])

            answer = ask_llm(context, question)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)