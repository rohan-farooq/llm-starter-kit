

import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss

# Load models once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

st.set_page_config(page_title="LLM Starter Kit", layout="wide")
st.title("🧠 LLM Starter Kit – Resume Analyzer, Summarizer & QA Bot")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Resume Analyzer", 
    "📝 Summarizer", 
    "🤖 QA Chatbot", 
    "📚 Chat with Your PDF (RAG)"
])

# ========== 📄 Resume Analyzer ==========
with tab1:
    st.header("📄 Resume Analyzer (Upload PDF)")
    uploaded_pdf = st.file_uploader("Upload your resume as a PDF", type=["pdf"])

    keywords = ["LLM", "NLP", "Python", "fine-tuning", "research", "project"]

    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.read()
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for page in doc:
                full_text += page.get_text()

            score = sum(word.lower() in full_text.lower() for word in keywords) * 10
            feedback = "✅ Strong resume!" if score >= 40 else "⚠️ Add more AI-related keywords and project outcomes."

            st.subheader("Extracted Resume Preview:")
            st.text_area("Text", full_text[:1000] + " ...", height=200)

            st.metric("Score", f"{score}/60")
            st.success(feedback)
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")

# ========== 📝 Summarizer ==========
with tab2:
    st.header("📝 Text Summarizer")
    long_text = st.text_area("Paste long text to summarize", height=300)
    if st.button("Summarize"):
        if long_text.strip():
            result = summarizer(long_text, max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary:")
            st.success(result[0]["summary_text"])
        else:
            st.warning("Please enter some text to summarize.")

# ========== 🤖 QA Chatbot ==========
with tab3:
    st.header("🤖 Question Answering Bot")
    context = st.text_area("Paste the context passage", height=200)
    question = st.text_input("Ask a question about the context")
    if st.button("Get Answer"):
        if context.strip() and question.strip():
            result = qa_pipeline(question=question, context=context)
            st.subheader("Answer:")
            st.success(result["answer"])
        else:
            st.warning("Please fill in both context and question.")

# RAG tab
with tab4:
    st.header("📚 Chat with Your PDF (RAG)")

    rag_file = st.file_uploader("Upload a PDF to chat with", type=["pdf"], key="rag")
    user_query = st.text_input("Ask a question about this PDF", key="rag_query")
    rag_button = st.button("Get Answer", key="rag_submit")

    if rag_file and rag_button:
        # Load PDF
        pdf_bytes = rag_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # Split into chunks
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

        # Create embeddings
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embed_model.encode(chunks)

        # Build FAISS index
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(embeddings)

        # Encode user question
        question_embedding = embed_model.encode([user_query])
        D, I = index.search(question_embedding, k=3)

        # Retrieve top chunks
        retrieved_context = "\n\n".join([chunks[i] for i in I[0]])

        # Use Hugging Face QA model
        result = qa_pipeline(question=user_query, context=retrieved_context)

        st.subheader("Answer:")
        st.success(result["answer"])

        with st.expander("🔍 Retrieved context used for this answer"):
            st.write(retrieved_context)
