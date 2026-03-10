# 🎙️ Voice AI . Let's Talk

A real-time, bidirectional voice conversation system that runs entirely in the browser. Built from scratch without using any managed voice AI platforms like LiveKit, Pipecat, Retell, or VAPI.

---

## Live Demo

- Frontend: Open `frontend/index.html` in any browser
- Backend: Run via Google Colab (see setup instructions below)

---

## Architecture

```
Browser (index.html)
        |
        |  PCM16 audio chunks  (binary WebSocket frames)
        |  JSON control events (text WebSocket frames)
        v
WebSocket Gateway (server.py)
        |
        v
VoicePipeline (pipeline.py)
        |
        |-- VAD       (numpy RMS thresholding)
        |-- STT       (Groq Whisper large-v3-turbo)
        |-- RAG       (optional - FAISS + sentence-transformers)
        |-- LLM       (Groq LLaMA 3.3 70B, streaming)
        |-- TTS       (gTTS - Google Text to Speech)
        |
        v
Browser Audio Playback (Web Audio API)
```

---

## How It Works

### 1. Audio Capture
The browser captures microphone input using `getUserMedia` with noise cancellation and echo suppression enabled. Audio is captured as raw PCM16 at 16kHz sample rate.

### 2. Streaming to Backend
Audio chunks are sent as binary WebSocket frames in real time. No batching, no base64 encoding. This keeps overhead minimal and latency low.

### 3. Voice Activity Detection
Two layers of VAD are used:

- **Backend VAD** - numpy RMS calculation detects end of speech after 700ms of silence and triggers the STT pipeline
- **Frontend VAD** - same RMS logic in JavaScript detects when user starts speaking while AI is talking and sends an interrupt signal immediately

### 4. Speech to Text
Groq Whisper large-v3-turbo converts the captured audio to text. Average latency is 300-400ms.

### 5. RAG (Optional)
If a knowledge base is loaded, FAISS vector search finds the most relevant context chunks using sentence-transformers embeddings. This context is injected into the LLM prompt.

### 6. LLM Response
Groq LLaMA 3.3 70B generates a response with streaming enabled. Responses are flushed sentence by sentence as soon as a punctuation boundary is detected, rather than waiting for the full response.

### 7. Text to Speech
Each sentence is immediately converted to MP3 audio using gTTS and streamed back to the browser via WebSocket binary frames.

### 8. Audio Playback
The browser receives MP3 chunks, combines them using the Web Audio API and plays them back in real time.

---

## Design Decisions

### Why WebSocket binary frames for audio
Raw PCM bytes sent as binary WebSocket frames avoid the 33% size overhead of base64 encoding and eliminate serialization latency.

### Why sentence level TTS flushing
Instead of waiting for the full LLM response before starting TTS, each sentence is sent to TTS as soon as it is complete. This means the first audio plays within 1 second of the LLM starting to respond rather than waiting for the entire reply.

### Why two layers of VAD
Backend VAD handles the normal end of speech detection. Frontend VAD handles interruptions instantly without a round trip to the server. When the user starts speaking while the AI is talking, the frontend immediately stops audio playback and sends an interrupt event.

### Why Groq instead of OpenAI
Groq provides free API access with extremely fast inference. Whisper large-v3-turbo on Groq is faster and cheaper than OpenAI Whisper for this use case.

---

## Latency Breakdown

| Stage | Typical Latency |
|---|---|
| VAD detection | less than 50ms |
| Whisper STT via Groq | 300 - 400ms |
| LLaMA 3.3 70B first token | 200 - 400ms |
| gTTS first audio | 300 - 500ms |
| Total end to end | 800 - 1400ms |

---

## Known Trade-offs

| Trade-off | Decision Made |
|---|---|
| Batch STT vs streaming STT | Using batch Whisper. Streaming STT like Deepgram would reduce latency by 200ms but requires a paid API |
| gTTS vs OpenAI TTS | gTTS is free but slightly robotic. OpenAI TTS-1 is more natural but requires billing |
| ScriptProcessor vs AudioWorklet | ScriptProcessor used for simplicity. AudioWorklet is the modern standard but more complex to implement |
| In memory RAG vs persistent DB | FAISS runs in process for speed. Not persistent across server restarts |
| Single process vs multi worker | Current server handles multiple concurrent sessions in one process. Production would need multiple workers |

---

## Tech Stack

| Component | Technology |
|---|---|
| WebSocket server | Python websockets library |
| Speech to Text | Groq Whisper large-v3-turbo |
| Language Model | Groq LLaMA 3.3 70B |
| Text to Speech | gTTS (Google Text to Speech) |
| Vector search | FAISS + sentence-transformers |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Tunnel | Cloudflare Tunnel |

---

## Project Structure

```
voice-ai/
├── backend/
│   ├── server.py          # WebSocket gateway and session handling
│   ├── pipeline.py        # STT + LLM + TTS orchestration
│   ├── session.py         # Session manager
│   ├── rag.py             # RAG retriever using FAISS
│   └── requirements.txt
├── frontend/
│   └── index.html         # Complete browser client
├── notebooks/
│   └── VoiceAI_SniperThink_Backend.ipynb
└── README.md
```

---

## Setup Instructions

### Option 1 - Google Colab (Recommended)

1. Open `notebooks/VoiceAI_SniperThink_Backend.ipynb` in Google Colab
2. Run all cells from top to bottom
3. Add your Groq API key when prompted
4. Copy the Cloudflare tunnel URL that is generated
5. Open `frontend/index.html` in your browser
6. Paste the tunnel URL and click Connect

### Option 2 - Local Setup

```bash
cd backend
pip install -r requirements.txt
pip install groq gtts

# Set environment variables
export GROQ_API_KEY=your_groq_api_key
export LLM_MODEL=llama-3.3-70b-versatile

# Start server
python server.py
```

Then open `frontend/index.html` in your browser and connect to `ws://localhost:8765`

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| GROQ_API_KEY | Your Groq API key | Required |
| LLM_MODEL | Groq model to use | llama-3.3-70b-versatile |
| TTS_VOICE | Voice for TTS | alloy |
| PORT | WebSocket server port | 8765 |

---

## RAG Usage

```python
from rag import RAGRetriever

retriever = RAGRetriever()
retriever.add_documents([
    {"id": "1", "text": "Your knowledge base content here."},
    {"id": "2", "text": "More content here."},
])

# Attach to pipeline
pipeline._rag_retriever = retriever
```

---

## Constraints Met

- No LiveKit
- No Pipecat
- No Daily.co
- No Retell AI
- No VAPI
- No Omnidim
- No Bolna
- All core voice infrastructure built from scratch

---

## Author

Pooja Yadav
GitHub: https://github.com/PoojaYadav-103/Voice_AI_Project
