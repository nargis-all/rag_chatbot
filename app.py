import os
import re
import traceback
import streamlit as st

from retriever import retrieve, count_word_occurrences
from llm import ask_llm
from upload_embeddings import upload_document

st.set_page_config(
    page_title="Customs RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Customs RAG Chatbot")
st.caption("Upload document(s) and ask questions about them.")

# ---------------- SESSION STATE ---------------- #

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

if "selected_files" not in st.session_state:
    st.session_state.selected_files = []

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("📂 Upload Document")

    uploaded_files = st.file_uploader(
        "Choose file(s)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("📤 Upload to Database"):

            os.makedirs("data", exist_ok=True)

            for uploaded_file in uploaded_files:

                save_path = os.path.join(
                    "data",
                    uploaded_file.name
                )

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    with st.spinner(f"Creating embeddings for {uploaded_file.name}..."):

                        total_chunks = upload_document(
                            save_path,
                            uploaded_file.name
                        )

                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)

                    st.success(
                        f"✅ {uploaded_file.name} uploaded! ({total_chunks} chunks)"
                    )

                except Exception as e:
                    st.error(f"Upload failed for {uploaded_file.name}: {e}")
                    st.code(traceback.format_exc())

    st.divider()

    if st.session_state.uploaded_files_list:
        st.subheader("📚 Select document(s) to search")

        st.session_state.selected_files = st.multiselect(
            "Documents",
            options=st.session_state.uploaded_files_list,
            default=st.session_state.uploaded_files_list
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

    if not st.session_state.selected_files:
        st.warning("📂 Please upload and select at least one document.")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    count_match = re.search(
        r"how many times.*?[\"']?(\w+)[\"']?\s+(?:is |was |be |)?(?:used|mentioned|appears?|occurs?)",
        question,
        re.IGNORECASE
    )

    if count_match:
        target_word = count_match.group(1)

        with st.spinner(f"Counting occurrences of '{target_word}'..."):
            total_count = sum(
                count_word_occurrences(f, target_word)
                for f in st.session_state.selected_files
            )

        st.subheader("💬 Answer")
        st.write(f"The word \"{target_word}\" appears **{total_count}** times across the selected document(s).")
        st.stop()

    try:
        with st.spinner("Searching document(s)..."):

            results = retrieve(
                question,
                file_names=st.session_state.selected_files,
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