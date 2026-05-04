# Manual tests

Start server:

```bash
uvicorn app:app --reload
```

Ingest:

```bash
curl -X POST http://127.0.0.1:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Ada Lovelace wrote notes about Charles Babbage’s Analytical Engine. She is often recognized for early ideas about computer programming.","source":"history-note"}'
```

Ask:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who wrote notes about the Analytical Engine?","top_k":3}'
```

Reset:

```bash
curl -X POST http://127.0.0.1:8000/reset
```
