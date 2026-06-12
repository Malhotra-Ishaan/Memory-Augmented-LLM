> 📌 **Note:** This repository was re-uploaded due to loss of access to the original repo's linked Git email account.

# 🧠 AI Memory Chat (Streamlit + Gemini + Vector Memory)

An intelligent Streamlit-based chat application that gives LLMs **long-term memory capabilities** using embeddings, FAISS retrieval, and NLI-based deduplication.

This project allows an AI assistant to remember user **facts and preferences across sessions**, making conversations more personalized and context-aware.

---

## 🚀 Features
- 💬 Chat interface using Streamlit
- 🧠 Persistent long-term memory system
- 🔎 Semantic search using Sentence Transformers
- ⚡ Fast vector retrieval using FAISS
- 🧹 Smart memory deduplication using:
  - Cosine similarity
  - NLI (Entailment / Contradiction detection)
- 📦 Local JSON-based storage (no database required)
- 🧾 Live memory viewer inside UI
- 🔄 New chat + clear memory controls

---

## 🏗️ System Architecture

### 1. Chat Layer
- Streamlit UI handles user interaction
- Stores conversation in `data.json`

### 2. Memory Layer
- Stores structured memory in `memory.json`
- Two memory types:
  - Preferences
  - Facts

### 3. Embedding Layer
- Model: `sentence-transformers/all-mpnet-base-v2`
- Converts text into dense vector embeddings

### 4. Retrieval Layer
- FAISS index for similarity search
- Hybrid scoring:
  - Cosine similarity
  - Importance score (frequency + recency)

### 5. Deduplication Layer
- Cross-encoder model:
  - `cross-encoder/nli-deberta-v3-base`
- Detects:
  - contradiction → replace memory
  - entailment → update frequency
  - neutral → add as new

### 6. LLM Layer
- Google Gemini model (`gemma-3-1b-it`)
- Injects relevant memory into prompts dynamically
