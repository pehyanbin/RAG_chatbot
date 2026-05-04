# Upgrade Notes

This version adds:

1. Chat UI
   - Open `/chat`
   - Upload files
   - Ask questions
   - View citations

2. Automatic document ingestion
   - New endpoint: `POST /ingest/files`
   - Supports PDF, TXT, MD, CSV, JSON, HTML

3. Improved search accuracy
   - New retrieval modes: `vector`, `keyword`, `hybrid`
   - New local embedding option: `local_sentence_transformers`

Recommended `.env`:

```env
LLM_PROVIDER=nvidia_nim_api
NVIDIA_API_KEY=your_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct

EMBEDDING_PROVIDER=local_sentence_transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_MODE=hybrid
```

After changing embedding provider:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/chat
```
