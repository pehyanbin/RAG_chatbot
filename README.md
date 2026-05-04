
# Multi-provider RAG Agent Starter

A small FastAPI RAG agent with:

- SQLite-backed vector/RAG database
- Document chunking and ingestion
- Retrieval with cosine similarity
- LLM provider router
- API-key providers:
  - OpenAI API key
  - OpenAI-compatible API
  - NVIDIA NIM API
  - Ollama API
  - OpenRouter API
  - Gemini API key
  - Anthropic API key
  - DeepSeek API
  - xAI API
  - Moonshot/Kimi API
- Local CLI/OAuth-style providers:
  - OpenAI Codex OAuth
  - OpenAI Plus / Pro subscription via Codex local sign-in
  - Gemini CLI OAuth
  - Google Antigravity OAuth-style local command bridge
  - OpenCode

> Important: ChatGPT Plus/Pro and CLI OAuth credentials are best treated as local interactive/dev tools, not production backend credentials. For a hosted backend, use API keys or a provider-specific OpenAI-compatible endpoint.

## Quick start

```bash
cd rag_agent_multi_provider
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Minimal test with no paid API

By default, the project uses:

- `LLM_PROVIDER=ollama_api`
- `OLLAMA_BASE_URL=http://localhost:11434/v1`
- `EMBEDDING_PROVIDER=hash`

Install Ollama, then run:

```bash
ollama pull llama3.2
ollama serve
```

Ingest text:

```bash
curl -X POST http://127.0.0.1:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text":"RAG means retrieval augmented generation. It retrieves relevant context before answering.","source":"notes"}'
```

Ask:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","top_k":3}'
```

## Configure provider

Set `LLM_PROVIDER` in `.env`.

Examples:

### OpenAI API key

```env
LLM_PROVIDER=openai_api_key
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
```

### OpenAI-compatible API

```env
LLM_PROVIDER=openai_compatible_api
OPENAI_COMPATIBLE_BASE_URL=https://your-provider.example.com/v1
OPENAI_COMPATIBLE_API_KEY=sk-...
OPENAI_COMPATIBLE_MODEL=your-model
```

### NVIDIA NIM API

```env
LLM_PROVIDER=nvidia_nim_api
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
```

### Ollama

```env
LLM_PROVIDER=ollama_api
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
```

### OpenRouter

```env
LLM_PROVIDER=openrouter_api
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4.1-mini
```

### Gemini API key

```env
LLM_PROVIDER=gemini_api_key
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

### Anthropic

```env
LLM_PROVIDER=anthropic_api_key
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

### DeepSeek

```env
LLM_PROVIDER=deepseek_api
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### xAI

```env
LLM_PROVIDER=xai_api
XAI_API_KEY=xai-...
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4
```

### Moonshot / Kimi

```env
LLM_PROVIDER=moonshot_ai
MOONSHOT_API_KEY=sk-...
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
MOONSHOT_MODEL=kimi-k2-0905-preview
```

## Embeddings

For serious RAG, switch away from the default hash embedder.

### OpenAI-compatible embeddings

```env
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
```

### Gemini embeddings

```env
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

If you change embedding models, delete `rag.sqlite3` and re-ingest your documents.

## Local CLI / OAuth bridge

These modes call a local CLI command. You must sign in locally first.

```env
LLM_PROVIDER=codex_oauth
CODEX_COMMAND_JSON=["codex","exec"]
```

```env
LLM_PROVIDER=gemini_cli_oauth
GEMINI_CLI_COMMAND_JSON=["gemini","-p"]
```

```env
LLM_PROVIDER=opencode
OPENCODE_COMMAND_JSON=["opencode","run"]
```

The app sends the final RAG prompt to the CLI through STDIN. Adjust the command JSON for your installed CLI version.

## Project layout

```text
app.py
rag_agent/
  agent.py
  chunking.py
  config.py
  embeddings.py
  llm.py
  vector_store.py
```

## Notes for production

- Put the database on managed storage or switch to Qdrant, Weaviate, Pinecone, LanceDB, or Postgres/pgvector.
- Add authentication to the FastAPI app.
- Add rate limits and audit logs.
- Do not store provider API keys in the database.
- Avoid using personal Plus/Pro/OAuth subscription sessions for server-side multi-user apps.


## Chat UI

Start the server:

```bash
uvicorn app:app --reload
```

Open the browser UI:

```text
http://127.0.0.1:8000/chat
```

The UI supports:

- ChatGPT-style message layout
- Multi-file upload
- Automatic document ingestion
- Source citations under each answer
- Retrieval mode selection: `hybrid`, `vector`, or `keyword`

## Better search accuracy

For quick tests, `EMBEDDING_PROVIDER=hash` works, but quality is limited.

Recommended local setting:

```env
EMBEDDING_PROVIDER=local_sentence_transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_MODE=hybrid
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

The first run downloads the embedding model. After changing embedding providers, reset and re-ingest documents:

```bash
curl -X POST http://127.0.0.1:8000/reset
```

Hybrid retrieval combines vector similarity with keyword matching, which usually improves results for names, codes, exact terms, and semantic questions.
