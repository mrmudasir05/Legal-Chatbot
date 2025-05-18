# 🧑‍⚖️ Pakistani Law Chatbot 🇵🇰

A smart Legal Assistant Chatbot built to answer **Pakistani law-related queries** using official legal documents — powered by **LLMs** and **RAG (Retrieval-Augmented Generation)**.

---

## 📌 Project Overview

This chatbot leverages **LLMs + Retrieval QA** to provide context-aware, accurate responses to user questions based on **uploaded Pakistani law PDFs**.

### 🔍 What It Does

- 💬 **Answers Legal Questions** using verified Pakistani law documents.
- 🔄 **RAG-based Pipeline**: Contextual responses via **LLM-powered QA**.
- 🛡️ **Secure Admin Panel**: Trusted users can upload and manage PDFs.
- 🌐 **User Interface**: Clean, interactive **Gradio** frontend.
- 🔐 **Environment-secure authentication** using `.env` for access control.

---

## 🛠️ Tech Stack

- 🐍 Python  
- 🧠 [LangChain](https://www.langchain.com/)  
- ⚡ [Groq LLM API](https://groq.com/)  
- 🔍 [FAISS](https://github.com/facebookresearch/faiss) for vector search  
- 🔡 SentenceTransformers for embeddings  
- 🖼️ Gradio for frontend interface  
- 🗂️ `.env` for secure credential storage

---

## 💡 What I Learned

- ✅ Real-world use of **Retrieval-Augmented Generation (RAG)**
- ✅ How to ingest and query **domain-specific documents**
- ✅ Building a **user-friendly, secure legal assistant**
- ✅ Role-based upload permissions using `.env`-controlled access

---

## 🔐 Admin Features

- Only **trusted users** (verified via environment password) can:
  - Upload new official legal PDFs
  - Update or remove outdated laws
- Ensures secure, controlled document management
