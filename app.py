# legal_lawyer_app.py
import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Constants
VECTOR_STORE_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt template
template = """
You are a legal assistant specialized in Pakistani law. Use the provided context to answer the legal query accurately.

Context:
{context}

Question:
{question}

Only answer based on the law. Do not hallucinate.
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Load Vectorstore
def load_vectorstore():
    if os.path.exists(VECTOR_STORE_DIR):
        return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

# Load initial vectorstore
vectorstore = load_vectorstore()
if not vectorstore:
    raise ValueError("❌ FAISS index not found. Please update laws first.")

# LLM
llm = ChatGroq(
    temperature=0,
    model="llama-3.1-8b-instant",
    max_tokens=1000
)

# QA chain
def get_qa_chain():
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

qa_chain = get_qa_chain()

# Function to handle legal question
def answer_question(query):
    if not query.strip():
        return "❌ Please enter a legal question."
    result = qa_chain.invoke(query)
    return result["result"]

# Function to rebuild FAISS vectorstore
def update_laws(pdf_files, password):
    if password != ADMIN_PASSWORD:
        return "❌ Invalid admin password."

    if not pdf_files:
        return "❌ No PDF files uploaded."

    docs = []
    for pdf in pdf_files:
        loader = PyMuPDFLoader(pdf.name)
        docs.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )   
    chunks = text_splitter.split_documents(docs)

    new_vectorstore = FAISS.from_documents(chunks, embeddings)

    # Replace old vectorstore
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
    new_vectorstore.save_local(VECTOR_STORE_DIR)

    # Reload in global chain
    global vectorstore, qa_chain
    vectorstore = load_vectorstore()
    qa_chain = get_qa_chain()

    return "✅ Laws updated successfully."

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🧑‍⚖️ **Pakistani Law Chatbot**")
    gr.Markdown("Ask legal questions based on the official law PDFs.")

    with gr.Tab("💬 Ask a Question"):
        text_input = gr.Textbox(label="Type your legal question", placeholder="e.g. What is the punishment for theft?")
        submit_btn = gr.Button("Get Legal Answer")
        answer_output = gr.Textbox(label="Legal Bot Answer")

        submit_btn.click(fn=answer_question, inputs=text_input, outputs=answer_output)

    with gr.Tab("🔐 Admin: Update Laws"):
        gr.Markdown("### Upload official law PDFs and update the knowledge base (Admin only).")

        pdf_upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload Law PDFs")
        password_input = gr.Textbox(label="Admin Password", type="password")
        update_button = gr.Button("Update Law Database")
        update_status = gr.Textbox(label="Status")

        update_button.click(fn=update_laws, inputs=[pdf_upload, password_input], outputs=update_status)

# Run app
if __name__ == "__main__":
    app.launch()
