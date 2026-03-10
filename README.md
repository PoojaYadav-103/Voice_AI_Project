# 🎙️ Real-Time Voice AI System – SniperThink Task

A complete, from-scratch, real-time bidirectional voice conversation system in the browser.
No LiveKit, no Pipecat, no Retell – every layer built explicitly.

---

## Architecture

```
Browser (index.html)
  │
  │  PCM16 audio chunks (binary WebSocket frames)
  │  JSON control events  (text WebSocket frames)
  ▼
WebSocket Gateway (server.py)
  │  per-connection session
  ▼
VoicePipeline (pipeline.py)
  ├── VAD  (numpy RMS thresholding)
  ├── STT  (OpenAI Whisper API)
  ├── RAG  (optional – FAISS + sentence-transformers)
  ├── LLM  (OpenAI GPT-4o-mini, streaming)
  └── TTS  (OpenAI TTS-1 → raw PCM → WebSocket)
```

### Data Flow (happy path)

```
Mic → PCM16 chunks → WebSocket → VAD
                                  │ speech detected
                                  ▼
                               Whisper STT  (100-400ms)
                                  │
                               RAG retrieval (optional, ~50ms)
                                  │
                               LLM streaming (first token ~200-400ms)
                                  │ sentence complete
                               TTS streaming (first audio ~300-500ms)
                                  │
                               WebSocket → Browser AudioContext playback
```

---

## Design Decisions

### 1. WebSocket binary framing
Raw PCM16 bytes are sent as WebSocket binary frames (not base64).
This eliminates ~33% overhead and reduces serialization latency.

### 2. Voice Activity Detection (VAD) – two layers
- **Backend VAD**: numpy RMS. Detects end-of-speech after 700ms silence → triggers STT.
- **Frontend VAD**: same RMS logic in JS. Detects user speaking while AI is talking → sends `interrupt` event immediately without a round-trip.

### 3. Sentence-level TTS streaming
Instead of waiting for the full LLM response, we flush TTS per sentence.
The first audio chunk plays ~300-500ms after the first sentence ends, not after the full reply.

### 4. Interruption handling
On `interrupt`:
- Frontend stops playback immediately (AudioContext re-init).
- Backend cancels the running asyncio TTS/LLM tasks.
- State resets to IDLE so the next utterance starts cleanly.

### 5. Conversation memory
Last 20 turns kept in memory per session. System prompt is always kept.

### 6. RAG (optional)
FAISS `IndexFlatIP` with normalized embeddings = cosine similarity.
`all-MiniLM-L6-v2` (22MB) embedded locally — no external vector DB.

---

## Latency Considerations

| Stage           | Typical Latency |
|----------------|----------------|
| VAD detection   | <50ms           |
| Whisper STT     | 100-400ms       |
| LLM first token | 200-400ms       |
| TTS first audio | 300-500ms       |
| **Total E2E**   | **700-1400ms**  |

### Optimizations applied
- PCM binary frames (no base64)
- Sentence-level TTS flushing (don't wait for full response)
- `tts-1` model (faster, lower quality) vs `tts-1-hd`
- `gpt-4o-mini` (fast, cheap) as default LLM
- `max_tokens=150` cap on LLM output

---

## Known Trade-offs

| Trade-off | Decision |
|-----------|----------|
| Whisper vs streaming STT | Whisper is batch; streaming STT (Deepgram/AssemblyAI) would reduce STT latency by 200-300ms but requires another API |
| ScriptProcessor vs AudioWorklet | ScriptProcessor is deprecated but simpler; AudioWorklet is the modern API |
| In-memory RAG vs persistent DB | FAISS in-process is fast but not persistent across restarts; add `.save()/.load()` for persistence |
| Single process vs multi-worker | Current server handles N concurrent sessions but is single-process; add gunicorn/uvicorn workers for production |

---

## Quickstart (Local)

```bash
# 1. Install backend deps
cd backend
pip install -r requirements.txt

# 2. Set env vars
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start server
python server.py

# 4. Open frontend
# Double-click frontend/index.html in your browser
# OR: python -m http.server 3000 (from frontend/ dir)

# 5. In the browser
# - URL: ws://localhost:8765
# - Click Connect
# - Click the mic orb and talk
```

---

## Quickstart (Google Colab)

Open `notebooks/VoiceAI_SniperThink_Backend.ipynb` in Google Colab.
Run cells 1-6. After Step 5 you get a public `wss://` URL.
Paste it into the frontend and click Connect.

---

## File Structure

```
voice-ai/
├── backend/
│   ├── server.py          # WebSocket gateway
│   ├── pipeline.py        # STT + LLM + TTS orchestration
│   ├── session.py         # Session manager
│   ├── rag.py             # RAG retriever (FAISS)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html         # Complete browser client (single file)
├── notebooks/
│   └── VoiceAI_SniperThink_Backend.ipynb
└── README.md
```

---

## Bonus: RAG Usage

```python
from rag import RAGRetriever, load_text_file

# From a text file
retriever = load_text_file("my_knowledge_base.txt", chunk_size=500)

# Or add docs manually
retriever = RAGRetriever()
retriever.add_documents([
    {"id": "1", "text": "Your product description here."},
    {"id": "2", "text": "FAQ content here."},
])

# Attach to pipeline
pipeline._rag_retriever = retriever
```
