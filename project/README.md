# 🤖 Context-Aware LangChain RAG Chatbot  
*A Modular Retrieval-Augmented Generation System built with LangChain, ChromaDB, and Multiple LLM Providers (OpenAI, Groq, Gemini)*  

[![LangChain](https://img.shields.io/badge/LangChain-0.3+-blue.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq API](https://img.shields.io/badge/Groq-LLM-orange.svg)](https://groq.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini-blue.svg)](https://deepmind.google/technologies/gemini/)

---

## 🧠 Overview

This repository contains a **Context-Aware LangChain RAG (Retrieval-Augmented Generation) Chatbot**, a modular and production-ready system designed for **document-grounded question answering**.  

The chatbot loads and processes your local documents (PDF, TXT, MD), embeds them using **Sentence Transformers**, stores them in **ChromaDB**, and uses **LLMs** (Groq, OpenAI, or Gemini) to generate **accurate, non-hallucinated answers**.

Developed as part of the **Ready Tensor AI Developer Certification Program (2025)**, this project demonstrates advanced RAG pipelines, structured prompt-building, and LLM safety best practices.

---

## 🎯 Problem it Solves

LLMs are often **not context-aware** and tend to **hallucinate** facts.  
This project solves that by integrating **retrieval-based context injection**, ensuring every answer is derived directly from trusted document sources.

---

## ⚙️ Key Features

✅ Retrieval-Augmented Generation (RAG) pipeline  
✅ Multi-LLM support (Groq, OpenAI, Gemini)  
✅ Structured prompt generation for consistency  
✅ Vector search with ChromaDB and Sentence Transformers  
✅ Document-grounded, anti-hallucination responses  
✅ Streamlit chat UI for real-time interaction  
✅ Prompt-injection and context leakage defense  

---

## 🧩 Project Structure

project/
    |___src/
    |   |__data/                      # Folder for user documents (PDF/TXT/MD)
    |   |   |__chatbotdoc.pdf         # Dependencies list
    |   │
    |   ├── app.py                    # Core RAG logic and orchestration
    |   ├── vectordb.py               # Embedding generation + ChromaDB store
    |   ├── prompt_builder.py         # Builds structured prompts dynamically
    |   ├── prompt_instructions.py    # System rules, tone, and safety constraints
    |   ├── run.py                    # Streamlit UI for interactive chatting
    |               
    |   
    |──.env                           # Environment variables (API keys, paths)
    |__.gitignore
    |__LICENSE
    |__README.md
    |__requirements.txt                

---

## 🧱 System Architecture

```mermaid
flowchart TD
    A[User Query] --> B[VectorDB Search (Chroma)]
    B --> C[Retrieve Top-k Relevant Chunks]
    C --> D[PromptBuilder Combines Context + Query]
    D --> E[LLM (Groq/OpenAI/Gemini) Generates Response]
    E --> F[Streamlit UI Displays Answer]
```

---

## ⚙️ How It Works

### Load Documents
- Reads PDFs or text files from `/data`.
- Splits text into semantic chunks with LangChain’s text splitters.

### Embed & Store
- Converts text chunks into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores embeddings in **ChromaDB**.

### Retrieve Context
- User queries trigger a similarity search to fetch the most relevant document chunks.

### Build Prompt
- **PromptBuilder** composes a structured system prompt using rules from **PromptInstructions**.
- Defines tone, safety, and reasoning constraints.

### Generate Answer
- The LLM (Groq, OpenAI, or Gemini) processes the context + user query.
- Produces document-grounded answers only.

### Display in Streamlit
- Chat-style UI for seamless user interaction.

---

## 🔐 Safety & Integrity Features

🛡️ **Instruction Shielding** – Blocks prompt-injection & reasoning exposure  
📘 **Context Restriction** – Answers strictly from retrieved document content  
🚫 **No Hallucination** – Declines politely when info is unavailable  
💬 **Response Control** – Maintains word count between 40–160  

Example fallback response:
> “I’m sorry, it seems that information isn’t covered in the document provided. Could you specify a topic or section to explore?”

---

## 🌍 Supported LLM Providers

| Provider | Package | Env Variables |
|-----------|----------|----------------|
| **Groq** | langchain-groq | `GROQ_API_KEY`, `GROQ_MODEL` |
| **OpenAI** | langchain-openai | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| **Google Gemini** | langchain-google-genai | `GOOGLE_API_KEY`, `GOOGLE_MODEL` |

---

## 🧩 Installation

```bash
# Clone the repository
git clone https://github.com/Richmondiroegbu/context-aware-rag-assistant.git
cd <src>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```ini
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
DATA_PATH=./data
```

Add your documents (e.g., *Agenda 2063: The Africa We Want – Popular Version.pdf*) into the `/data` folder.

---

## ▶️ Running the App

Run the Streamlit interface:

```bash
streamlit run run.py
```

Then open your browser to:  
👉 **http://localhost:8501**
   **http://192.168.0.187:8501**
    

---

## 💬 Example Chat

**User:**  
> What are the main goals of Agenda 2063?

**Assistant:**  
> According to the document, Agenda 2063 envisions a prosperous and unified Africa driven by inclusive growth, sustainable development, and regional integration, promoting peace and global influence.

---


## 🧩 Future Enhancements

🧠 Add memory for multi-turn contextual chats  
🌐 Real-time WebSocket streaming for faster response  
📤 Upload documents directly via Streamlit  
🧮 Add retrieval evaluation metrics (precision / recall)  
💻 Integrate local LLMs (Ollama, LM Studio) for offline inference  

---

## 🧑‍💻 Author

**Richmond Iroegbu**  
*Data Science & Machine Learning Engineer*  
📜 Developed for the **Ready Tensor AI Developer Certification Program (2025)**  
🔗 *LinkedIn · GitHub*

---

## 🪪 License

This project is licensed under the **MIT License** — free for personal and commercial use.  
See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **LangChain** – Modular framework for LLM apps  
- **ChromaDB** – Vector store and retrieval engine  
- **Sentence Transformers** – For text embeddings  
- **Streamlit** – Frontend interface  
- **Groq API** – High-performance inference provider  
- **Google Gemini** – Multi-modal LLM  

💡 This repository demonstrates a reliable, ethical, and scalable LangChain-based RAG architecture designed for real-world AI applications.

