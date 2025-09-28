# 🩺 Medical Chatbot

A Flask-based **medical chatbot** powered by **LangChain**, **Hugging Face LLMs**, and **FAISS** for retrieval-augmented generation (RAG).  
It can answer medical-related questions using a combination of PDFs, CSV data, and embeddings.

---

## 🚀 Features
- Flask web app with a simple chat interface  
- Hugging Face `zephyr-7b-beta` LLM integration  
- Retrieval-Augmented Generation (RAG) using FAISS vectorstore  
- PDF and CSV document ingestion  
- Custom prompt template for safe medical answers  
- Dockerized for deployment  
- GitHub Actions workflow to build & push Docker images automatically to GitHub Container Registry (GHCR)  

---

## 📂 Project Structure


---

## ⚙️ Installation (Local)

1. Clone the repo:
   ```bash
   git clone   https://github.com/okefemi12/medical-chatbot.git
   cd medical-chatbot

2. python -m venv my-env
   source my-env/bin/activate   # macOS/Linux
   my-env\Scripts\activate      # Windows

3. pip install -r requirements.txt
4. HF_TOKEN=your_huggingface_api_key
5. python app.py



