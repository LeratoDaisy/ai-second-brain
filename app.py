# app.py
import os
import streamlit as st
from ingest import process_file
from query import ask_question

# --- Streamlit App Config ---
st.set_page_config(
    page_title="AI Second Brain",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 AI Second Brain")
st.write(
    "Upload PDFs, and ask questions — powered by OpenAI embeddings and GPT."
)

# --- Sidebar: Upload PDF ---
st.sidebar.header("Upload your document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded PDF temporarily
    temp_path = os.path.join("vectorstore", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process PDF: chunk + embeddings
    with st.spinner("Processing PDF..."):
        process_file(temp_path)
    st.success(f"Processed {uploaded_file.name} successfully!")

# --- Ask a question ---
st.sidebar.header("Ask a question")
question = st.sidebar.text_input("Your question:")

if st.sidebar.button("Ask") and question:
    if not uploaded_file:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Getting answer..."):
            answer = ask_question(question)
        st.write("**Answer:**")
        st.write(answer)