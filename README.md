# 🧠 SHAKAL — STUDY MADAD

### *Your AI Study Assistant*

> Upload your study material — PDFs, websites, YouTube lectures, even images — and chat with an AI that teaches it back to you, your way.

🔗 **Live Demo:** [shakal-gaze7idvxd3bnaf6f3zbdu.streamlit.app](https://shakal-gaze7idvxd3bnaf6f3zbdu.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-00A3E0?style=for-the-badge)
![Mistral AI](https://img.shields.io/badge/Powered%20by-Mistral%20AI-FF7000?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-8B5CF6?style=for-the-badge)

---

## ✨ Features

- 📚 **Multi-Source RAG** — feed SHAKAL up to any number of sources at once:
  - 📄 **PDFs** — notes, books, papers
  - 🌐 **Websites** — any public URL
  - 🎥 **YouTube** — automatic transcript extraction
  - 🖼️ **Images** — text extracted via **Mistral OCR** with a **Vision-LLM fallback**
- 🎭 **5 Teaching Personalities** — choose *how* you want to learn
- 💬 **Source-Grounded Chat** — answers prioritize *your* material, with clear notices when general knowledge is used
-  **Smart Retrieval** — MMR-based semantic search over ChromaDB embeddings
- ⚡ **Modern Tech UI** — dark glassmorphism theme, neon gradients, animated processing states with rotating AI quotes & fun facts
- 🖥️ **Two Interfaces** — sleek Streamlit web app + a terminal CLI version

---

## 🎭 Personalities

| Personality | Teaching Style |
|---|---|
| **Normal** | Balanced, clear explanations with examples & analogies |
| **Theorist** | Deep conceptual foundations — *why* things work |
| **Practicalist** | Real-world applications & implementation |
| **Examiner** | Strict evaluation, exam focus, mistake analysis |
| **Guide** | Socratic mentor — builds independent problem-solving |

---

## 🏗️ How It Works

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│   SOURCES   │──▶│   LOADERS    │──▶│  CHUNKING   │──▶│  ChromaDB    │
│ PDF/WEB/YT/ │   │ PyPDF, BS4,  │   │ Recursive   │   │  Embeddings  │
│   IMAGE     │   │ YT, OCR+LLM  │   │ TextSplitter│   │ (MiniLM-L6)  │
└─────────────┘   └──────────────┘   └─────────────┘   └──────┬───────┘
                                                              │
┌─────────────┐   ┌──────────────┐   ┌─────────────┐          │
│   ANSWER    │◀──│  Mistral     │◀──│   Prompt    │◀─────────┘
│  (chat UI)  │   │  Small 2506  │   │ + MMR Search│   MMR Retrieval
└─────────────┘   └──────────────┘   └─────────────┘
~~~

1. **Ingest** — sources are loaded, cleaned and split into chunks.
2. **Embed** — chunks are vectorized with `all-MiniLM-L6-v2` and stored in ChromaDB.
3. **Retrieve** — your question is matched against chunks using **MMR** (balanced relevance + diversity).
4. **Generate** — Mistral Small 2506 answers using your sources + the active personality prompt.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (custom CSS theme) |
| LLM | Mistral AI — `mistral-small-2506` |
| Image OCR | Mistral OCR (`mistral-ocr-latest`) + Vision fallback |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Loaders | PyPDFLoader, WebBaseLoader, youtube-transcript-api |

---

## 📦 Installation (Local)

```bash
# 1. Clone the repo
git clone https://github.com/aniket-andyy/shakal-study-madad.git
cd shakal-study-madad

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "MISTRAL_API_KEY=your_key_here" > .env

# 4. Run the web app
streamlit run app.py

# ...or run the CLI version
python main.py
```

### 🔑 Environment Variables

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Your [Mistral AI](https://console.mistral.ai/) API key |

> On **Streamlit Cloud**, add it under *App Settings → Secrets* instead of `.env`.

---

## 📁 Project Structure

~~~
├── app.py            #  Streamlit web app (modern UI)
├── main.py           # 🖥️ Terminal CLI version (3 working modes)
├── database.py       # 🗄️ Loaders, OCR, chunking & ChromaDB ingestion
├── requirements.txt  # 📦 Python dependencies
└── README.md         # 📖 You are here
```

---

## 🖼️ Screenshots

<!-- Add your screenshots here, e.g.:
![Home](screenshots/home.png)
![Chat](screenshots/chat.png)
-->

*Coming soon — drop your screenshots in a `/screenshots` folder and link them above.*

---

## 🗺️ Roadmap

- [ ] Chat history persistence
- [ ] Source citations in answers
- [ ] Multi-language support (Hindi + English)
- [ ] Voice input / TTS answers

---

## 🙌 Developed By

**Aniket Sharma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aniket%20Sharma-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aniket-sharma-42a700418)
[![GitHub](https://img.shields.io/badge/GitHub-aniket--andyy-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aniket-andyy)

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

<p align="center">
  Made with 💙 and a lot of ☕ by <b>Aniket Sharma</b>
  <br/>
  <i>“ aniket bhai ki taraf se hello! :) ”</i>
</p> 
