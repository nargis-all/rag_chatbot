import os
import streamlit as st
from retriever import retrieve
from llm import ask_llm
from upload_embeddings import upload_document

st.set_page_config(
    page_title="Customs RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Customs RAG Chatbot")
st.caption("Ask questions about customs regulations using AI.")

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a DOCX file",
        type=["docx"]
    )

    if uploaded_file:

        os.makedirs("data", exist_ok=True)

        save_path = os.path.join("data", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("📤 Upload to Database"):

            with st.spinner("Creating embeddings..."):

                total_chunks = upload_document(
                    save_path,
                    uploaded_file.name
                )

            st.success(f"✅ Uploaded successfully! ({total_chunks} chunks)")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.clear()
        st.rerun()

# ---------------- Main ---------------- #

st.subheader("💬 Ask a Question")

question = st.text_input(
    "",
    placeholder="Example: What is temporary import?"
)

if st.button("🔍 Get Answer", use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching documents..."):

        results = retrieve(
    question,
    uploaded_file.name,
    top_k=3
)

        if not results:
            st.error("No relevant information was found.")
            st.stop()

        context = "\n\n".join(
            doc["content"][:800]
            for doc in results
        )

        answer = ask_llm(context, question)

    st.subheader("💬 Answer")
    st.write(answer)

    with st.expander("📄 Retrieved Context"):
        st.write(context)
    with st.expander("📄 Retrieved Context"):
        st.write(context)