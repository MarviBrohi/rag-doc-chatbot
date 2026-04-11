<div align="center">

# 🧠 DocuMind AI

### *Chat With Your Documents Using Artificial Intelligence*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.5-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **DocuMind AI** is a production-grade **RAG (Retrieval-Augmented Generation)** chatbot that lets you upload any PDF document and have an intelligent conversation with its content. It retrieves the most relevant sections from your documents and generates accurate, cited answers using LLaMA3 — all in a sleek dark-themed UI.

<br/>

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

</div>

## 📸 Demo

> *Upload a PDF → Ask questions → Get cited answers instantly*

```
User: "What are the key findings in this research paper?"

🧠 DocuMind AI:
Based on the document, the key findings are:

• Finding 1 — [Source: research_paper.pdf, Page 3]
• Finding 2 — [Source: research_paper.pdf, Page 7]
• Finding 3 — [Source: research_paper.pdf, Page 12]

📎 View Sources → Page 3 | Page 7 | Page 12
```

---

## ✨ Features

| Feature | Description |
|:---|:---|
| 📄 **Multi-PDF Upload** | Upload and query multiple PDF documents simultaneously |
| 🧠 **RAG Pipeline** | Retrieval-Augmented Generation for highly accurate answers |
| 📎 **Source Citations** | Every answer includes exact page number and source references |
| ⚡ **Ultra-Fast Inference** | Powered by Groq's LPU — fastest LLM inference available |
| 💬 **Typing Animation** | Smooth character-by-character response for great UX |
| 🎨 **Professional Dark UI** | Custom dark theme with gradient accents and animations |
| 📊 **Live Session Stats** | Real-time tracking of documents, chunks, and messages |
| 💡 **Sample Questions** | One-click sample prompts to get started instantly |
| 🗑️ **Session Reset** | Clear everything and start fresh with one click |
| 🔒 **Secure API Handling** | API keys via `.env` and Streamlit Secrets — never hardcoded |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                    Streamlit Dark UI                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING                       │
│                                                              │
│  PDF Upload → PyMuPDF Parser → Text Extraction               │
│       → RecursiveCharacterTextSplitter (800 chars)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING & STORAGE                       │
│                                                              │
│  HuggingFace all-MiniLM-L6-v2 → Vector Embeddings           │
│               → FAISS Vector Database                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG RETRIEVAL ENGINE                       │
│                                                              │
│  User Query → Semantic Search → Top-4 Relevant Chunks        │
│            → Context Builder → Prompt Engineering            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM ANSWER GENERATION                     │
│                                                              │
│         Groq API → LLaMA3-8B-8192 → Cited Answer            │
│              → Streamlit Chat UI → User                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:---:|:---:|:---|
| 🎨 **Frontend** | Streamlit + Custom CSS | Interactive dark-themed UI |
| 🤖 **LLM** | Groq API (LLaMA3-8B) | Ultra-fast answer generation |
| 🔢 **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Convert text to vectors |
| 🗄️ **Vector DB** | FAISS (Facebook AI) | Store & search embeddings |
| ⛓️ **RAG Framework** | LangChain 0.2.5 | Orchestrate the RAG pipeline |
| 📄 **PDF Parser** | PyMuPDF (fitz) | Extract text + metadata from PDFs |
| 🧠 **ML Framework** | PyTorch + Transformers | Power the embedding model |
| 🔐 **Secrets** | python-dotenv | Secure API key management |

</div>

---

## 📁 Project Structure

```
rag-doc-chatbot/
│
├── 📄 app.py                    ← Main Streamlit application & UI
│
├── 📂 rag/                      ← Core RAG pipeline
│   ├── 🐍 __init__.py
│   ├── 🐍 embedder.py           ← Text chunking + FAISS embedding
│   ├── 🐍 retriever.py          ← Semantic similarity search
│   └── 🐍 llm_chain.py         ← Groq LLaMA3 answer generation
│
├── 📂 utils/                    ← Helper utilities
│   ├── 🐍 __init__.py
│   └── 🐍 pdf_parser.py        ← PDF text & metadata extraction
│
├── 📂 .streamlit/               ← Streamlit configuration
│   ├── ⚙️ config.toml           ← Dark theme settings
│   └── 🔒 secrets.toml         ← API keys (NOT on GitHub)
│
├── 📝 .env.example              ← Environment variable template
├── 🚫 .gitignore                ← Protects secrets from GitHub
├── 📦 requirements.txt          ← All Python dependencies
└── 📖 README.md                 ← You are here!
```

---

## ⚙️ Local Setup & Installation

### Prerequisites

- Python 3.10 or higher
- 5 GB free disk space (for PyTorch + packages)
- Free Groq API key → [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-doc-chatbot.git
cd rag-doc-chatbot
```

### Step 2 — Create virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure API key

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🚀

---

## 🌐 Deploy to Streamlit Cloud

```
1. Push code to GitHub
2. Visit → share.streamlit.io
3. Connect your GitHub repo
4. Set main file → app.py
5. Go to Settings → Secrets → Add:
        GROQ_API_KEY = "gsk_your_key_here"
6. Click Deploy!
```

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 💡 How to Use

```
Step 1 → Upload one or more PDF files in the sidebar
Step 2 → Click ⚡ Process to index the documents
Step 3 → Ask any question in the chat input
Step 4 → Click 📎 View Sources to see exact page references
Step 5 → Try sidebar sample questions for quick demos
Step 6 → Click 🗑️ Clear to reset the session
```

---

## 🧠 How RAG Works

**RAG = Retrieval-Augmented Generation**

Traditional LLMs answer from training data only — they hallucinate when asked about your private documents. RAG solves this:

```
Without RAG:  Question → LLM → Guessed Answer ❌
With RAG:     Question → Search Your Docs → Relevant Chunks → LLM → Accurate Answer ✅
```

**The 4 stages in DocuMind AI:**

1. **Index** — PDF text is split into 800-character chunks and converted to vector embeddings using HuggingFace's `all-MiniLM-L6-v2` model
2. **Store** — All vectors are stored in a FAISS index for lightning-fast similarity search
3. **Retrieve** — When you ask a question, the top 4 most semantically similar chunks are retrieved
4. **Generate** — Your question + retrieved context is sent to Groq LLaMA3, which generates a cited, accurate answer

---

## 📊 Performance

| Metric | Value |
|:---|:---|
| ⚡ Average Response Time | ~1–2 seconds |
| 📄 Max PDF Size | 200 MB |
| 🔢 Chunk Size | 800 characters |
| 🔍 Chunks Retrieved | Top 4 per query |
| 🤖 LLM Model | LLaMA3-8B-8192 |
| 🌡️ Temperature | 0.3 (precise answers) |

---

## 🔮 Roadmap

- [x] Multi-PDF support
- [x] Source citations with page numbers
- [x] Professional dark UI
- [x] Typing animation effect
- [ ] Export chat history as PDF
- [ ] Multi-language document support
- [ ] Upgrade to LLaMA3-70B model
- [ ] Document comparison mode
- [ ] Voice input support

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the project
2. Create your feature branch → `git checkout -b feature/amazing-feature`
3. Commit changes → `git commit -m 'Add amazing feature'`
4. Push to branch → `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">

## 👨‍💻 Author

**[Your Name]**
*AI/ML Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)
[![Gmail](https://img.shields.io/badge/Gmail-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@gmail.com)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Built with ❤️ using Python, LangChain, and Groq*

</div>
