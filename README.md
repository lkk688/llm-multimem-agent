
# Multimodal Memory LLM and AI Agent System

This project provides a modular framework to build a **Multimodal Memory-Augmented LLM system** with:

- 🧠 FAISS-based vector memory
- 🖼️ Multimodal embedding (Text, Image, Audio)
- 🔌 Model Context Protocol (MCP) server for context injection
- 🤖 LLM backend support: OpenAI, vLLM, Ollama, litellm
- 🧰 Tool Calling and Prompt Orchestration (MCP plugins)
- 🌐 UI options: Gradio app, FastAPI endpoints, optional React frontend
- 📘 Documentation: MkDocs + Jupyter notebook rendering

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Memory Search (FAISS)** | Store & retrieve embeddings across modalities |
| **Multimodal Embedding** | Text (SentenceTransformer), Image (CLIP), Audio (Whisper/Wav2Vec) |
| **RAG-ready Prompting** | Retrieve relevant memory and inject into LLM prompt |
| **Model Context Protocol (MCP)** | Standardized JSON-based context injection |
| **Tool Calling** | Extend LLM with function calls (e.g., weather, web search) |
| **Multiple UIs** | Gradio + optional React-based frontend via API |
| **MkDocs** | Documentation site with notebook integration |

---

## 🛠 Installation

```bash
# Clone and install
git clone https://github.com/lkk688/llm-multimem-agent.git
cd llm-multimem-agent
pip install -e .
```

---

## 🧪 Quick Start

### 1. Start MCP Context Server

```bash
uvicorn multimem.mcp.server:app --port 8000
```

### 2. Launch Gradio UI

```bash
python -m multimem.ui.gradio_app
```

---

## 🧠 LLM Backends

Supports OpenAI, Ollama, vLLM multiple LLM backend.

To utilize the latest Responses api from OpenAI, upgrade the openai package:
```bash
pip install --upgrade openai
```
OpenAI LLM model utility file is `llmmultimem/llm/openai_utils.py`


You can specify the backend and model in the Gradio UI or in your own app.

---

## 🧩 MCP Context Server

Send POST requests to `/mcp/context` with your `user_input`, and receive memory+tool-aware prompts. See `mcp.protocol.py` for schema.

---

## 🌐 UI Options

### ✅ Gradio (default)
Easy to use, no frontend setup required.

### ✅ FastAPI endpoints
Use with your own app or React frontend.

### ✅ React-based frontend
Use FastAPI as backend and build a web UI with:

- React + Tailwind + shadcn/ui
- Call `/mcp/context` and `/chat` APIs
- (Sample repo coming soon!)

---

## 📚 Documentation

### Build local docs

```bash
pip install mkdocs-material mkdocs-jupyter
mkdocs serve
```

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

---

## 📓 Notebooks

Interactive notebooks available in `docs/notebooks/`, rendered in documentation.

---

## 📦 Project Structure

```
multimem/
├── ui/                # Gradio app
├── llm/               # LLM backend support
├── mcp/               # Model Context Protocol server
├── memory/            # FAISS-based memory
├── embedder/          # Text, image, audio embedding
├── config.py          # Global config
docs/                  # MkDocs site (Markdown + Notebooks)
setup.py, mkdocs.yml, requirements.txt, pyproject.toml
```

---

## 📝 License

MIT License.
