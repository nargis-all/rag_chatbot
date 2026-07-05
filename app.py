import os
import traceback
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
st.caption("Upload a document and ask questions about it.")

# ---------------- SESSION STATE ---------------- #

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:

        if st.button("📤 Upload to Database"):

            os.makedirs("data", exist_ok=True)

            save_path = os.path.join(
                "data",
                uploaded_file.name
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Creating embeddings..."):

                total_chunks = upload_document(
                    save_path,
                    uploaded_file.name
                )

            st.session_state.current_file = uploaded_file.name

            st.success(
                f"✅ Uploaded successfully! ({total_chunks} chunks)"
            )

    st.divider()

    if st.session_state.current_file:
        st.info(
            f"Current document:\n{st.session_state.current_file}"
        )

    if st.button("🗑 Clear Chat"):
        st.session_state.clear()
        st.rerun()

# ---------------- MAIN ---------------- #

st.subheader("💬 Ask a Question")

question = st.text_input(
    "",
    placeholder="Example: What is temporary import?"
)

if st.button("🔍 Get Answer", use_container_width=True):

    if st.session_state.current_file is None:
        st.warning("📂 Please upload a document first.")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Searching document..."):

            results = retrieve(
                question,
                st.session_state.current_file,
                top_k=3
            )

            if not results:
                st.error("No relevant information found.")
                st.stop()

            context = "\n\n".join(
                doc["content"][:800]
                for doc in results
            )

            answer = ask_llm(
                context,
                question
            )

        st.subheader("💬 Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Context"):
            st.write(context)

    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())