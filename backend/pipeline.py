"""
Voice Pipeline
Orchestrates: Audio Buffer → STT → LLM → TTS → WebSocket playback
Handles interruptions, turn-taking, and streaming at each stage.
"""

import asyncio
import io
import json
import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000          # Hz – input audio
CHUNK_DURATION_MS = 20       # ms per audio chunk
SILENCE_THRESHOLD = 0.01     # RMS below this = silence
SILENCE_DURATION_MS = 700    # ms of silence → end of speech
VAD_WINDOW_CHUNKS = int(SILENCE_DURATION_MS / CHUNK_DURATION_MS)  # 35 chunks


class VoicePipeline:
    """
    End-to-end voice AI pipeline for one session.

    States:
        IDLE       – waiting for speech
        LISTENING  – user is speaking
        PROCESSING – running STT + LLM + TTS
        SPEAKING   – streaming TTS audio back
    """

    def __init__(self, session_id: str, websocket):
        self.session_id = session_id
        self.ws = websocket

        # Audio accumulation
        self._audio_buffer: bytes = b""
        self._silence_chunks = 0
        self._is_user_speaking = False
        self._speech_start_time: Optional[float] = None

        # Pipeline state
        self._state = "IDLE"
        self._processing_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None

        # Conversation history for LLM
        self._conversation: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, concise voice assistant. "
                    "Keep responses SHORT (1-3 sentences) since they will be spoken aloud. "
                    "Do not use markdown, bullet points, or special characters."
                ),
            }
        ]

        # Lazy-load clients (set on first use)
        self._openai_client = None
        self._rag_retriever = None

        logger.info(f"[{self.session_id}] Pipeline initialized")

    # ──────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────

    async def handle_audio_chunk(self, chunk: bytes):
        """Receive raw PCM16 audio from the browser."""
        rms = self._compute_rms(chunk)
        is_speech = rms > SILENCE_THRESHOLD

        if is_speech:
            self._silence_chunks = 0
            if not self._is_user_speaking:
                self._is_user_speaking = True
                self._speech_start_time = time.time()
                self._audio_buffer = b""  # fresh buffer
                await self._send_event("speech_started")
                logger.info(f"[{self.session_id}] Speech started (RMS={rms:.4f})")

                # Interrupt AI if it was speaking
                if self._state == "SPEAKING":
                    await self.handle_interrupt()

            self._audio_buffer += chunk

        else:
            if self._is_user_speaking:
                self._silence_chunks += 1
                self._audio_buffer += chunk  # keep trailing silence

                if self._silence_chunks >= VAD_WINDOW_CHUNKS:
                    # End-of-speech detected by VAD
                    await self.handle_end_of_speech()

    async def handle_end_of_speech(self):
        """Called when user stops speaking (VAD or explicit signal)."""
        if not self._is_user_speaking and not self._audio_buffer:
            return

        self._is_user_speaking = False
        duration_ms = (
            (time.time() - self._speech_start_time) * 1000
            if self._speech_start_time
            else 0
        )
        logger.info(
            f"[{self.session_id}] End of speech. "
            f"Duration={duration_ms:.0f}ms  Buffer={len(self._audio_buffer)} bytes"
        )

        if len(self._audio_buffer) < 3200:  # < 100ms of audio → skip
            logger.info(f"[{self.session_id}] Audio too short, ignoring")
            self._audio_buffer = b""
            return

        audio_snapshot = self._audio_buffer
        self._audio_buffer = b""
        self._state = "PROCESSING"

        await self._send_event("processing_started")
        self._processing_task = asyncio.create_task(self._run_pipeline(audio_snapshot))

    async def handle_interrupt(self):
        """User interrupted the AI while it was speaking."""
        logger.info(f"[{self.session_id}] Interrupt received")
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        self._state = "IDLE"
        await self._send_event("interrupted")

    async def cleanup(self):
        """Cancel all running tasks."""
        for task in [self._processing_task, self._tts_task]:
            if task and not task.done():
                task.cancel()

    # ──────────────────────────────────────────
    # Internal pipeline stages
    # ──────────────────────────────────────────

    async def _run_pipeline(self, audio: bytes):
        """STT → (RAG) → LLM → TTS streaming."""
        try:
            t0 = time.time()

            # 1. Speech-to-Text
            transcript = await self._transcribe(audio)
            if not transcript:
                self._state = "IDLE"
                return
            logger.info(f"[{self.session_id}] STT ({(time.time()-t0)*1000:.0f}ms): '{transcript}'")
            await self._send_event("transcript", {"text": transcript})

            # 2. Optional RAG context injection
            rag_context = ""
            if self._rag_retriever:
                rag_context = await self._retrieve_context(transcript)

            # 3. LLM (streaming)
            self._state = "SPEAKING"
            self._tts_task = asyncio.create_task(
                self._llm_and_tts(transcript, rag_context, stt_latency_ms=(time.time()-t0)*1000)
            )
            await self._tts_task

        except asyncio.CancelledError:
            logger.info(f"[{self.session_id}] Pipeline cancelled")
        except Exception as e:
            logger.error(f"[{self.session_id}] Pipeline error: {e}", exc_info=True)
            await self._send_event("error", {"message": str(e)})
        finally:
            self._state = "IDLE"

    async def _transcribe(self, audio: bytes) -> str:
        """
        Convert PCM16 audio to text using Whisper API.
        Falls back to a stub if no API key configured.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OPENAI_API_KEY – using stub transcription")
            return "[stub transcript – set OPENAI_API_KEY]"

        try:
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)

            # Convert raw PCM to WAV bytes in memory
            wav_bytes = _pcm_to_wav(audio, sample_rate=SAMPLE_RATE)
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"

            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
            return response.strip()
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""

    async def _retrieve_context(self, query: str) -> str:
        """RAG retrieval – returns relevant context string."""
        try:
            docs = self._rag_retriever.retrieve(query, top_k=3)
            if docs:
                context = "\n\n".join(d["text"] for d in docs)
                return f"\n\nRelevant context:\n{context}\n"
        except Exception as e:
            logger.error(f"RAG error: {e}")
        return ""

    async def _llm_and_tts(self, user_text: str, rag_context: str, stt_latency_ms: float):
        """
        Stream LLM tokens, accumulate sentences, pipe each sentence to TTS
        and stream audio chunks back to the browser.
        Goal: first audio playback < 1 second after STT finishes.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            stub = "I am a stub response. Please set your OpenAI API key."
            await self._stream_tts_sentence(stub)
            return

        import openai
        client = openai.AsyncOpenAI(api_key=api_key)

        # Build message list with optional RAG context
        messages = list(self._conversation)
        user_content = user_text
        if rag_context:
            user_content = f"{rag_context}\n\nUser question: {user_text}"
        messages.append({"role": "user", "content": user_content})

        full_response = ""
        sentence_buffer = ""
        first_audio_sent = False
        t_llm_start = time.time()

        try:
            stream = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=messages,
                stream=True,
                max_tokens=150,
                temperature=0.7,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                sentence_buffer += delta

                # Flush on sentence boundaries for low first-word latency
                flush_chars = {".", "!", "?", "\n"}
                if any(c in sentence_buffer for c in flush_chars):
                    sentences = _split_sentences(sentence_buffer)
                    for sentence in sentences[:-1]:  # keep last partial
                        sentence = sentence.strip()
                        if sentence:
                            if not first_audio_sent:
                                latency = (time.time() - t_llm_start) * 1000
                                logger.info(
                                    f"[{self.session_id}] First sentence latency: "
                                    f"STT={stt_latency_ms:.0f}ms LLM={latency:.0f}ms"
                                )
                                first_audio_sent = True
                            await self._stream_tts_sentence(sentence)
                    sentence_buffer = sentences[-1] if sentences else ""

            # Flush remaining
            if sentence_buffer.strip():
                await self._stream_tts_sentence(sentence_buffer.strip())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            return

        # Save assistant reply to conversation history
        if full_response:
            self._conversation.append({"role": "user", "content": user_text})
            self._conversation.append({"role": "assistant", "content": full_response})
            # Keep last 20 turns to avoid token overflow
            if len(self._conversation) > 21:
                self._conversation = self._conversation[:1] + self._conversation[-20:]

        await self._send_event("response_complete", {"text": full_response})

    async def _stream_tts_sentence(self, text: str):
        """Convert one sentence to speech and stream PCM audio to the browser."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return

        try:
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)

            await self._send_event("tts_start", {"text": text})

            async with client.audio.speech.with_streaming_response.create(
                model="tts-1",          # tts-1 is faster; tts-1-hd is higher quality
                voice=os.getenv("TTS_VOICE", "alloy"),
                input=text,
                response_format="pcm",  # raw PCM16 @ 24kHz – no decoding needed
            ) as response:
                async for audio_chunk in response.iter_bytes(chunk_size=4096):
                    if asyncio.current_task().cancelled():
                        return
                    # Send raw PCM bytes directly – browser plays them
                    await self.ws.send(audio_chunk)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TTS error for '{text[:30]}…': {e}")

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _compute_rms(pcm_bytes: bytes) -> float:
        """Root Mean Square of PCM16 audio – used for VAD."""
        if len(pcm_bytes) < 2:
            return 0.0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(samples ** 2)) / 32768.0)

    async def _send_event(self, event_type: str, data: Optional[dict] = None):
        """Send a JSON control event to the browser."""
        try:
            payload = {"type": event_type}
            if data:
                payload.update(data)
            await self.ws.send(json.dumps(payload))
        except Exception:
            pass  # connection may already be closed


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    import struct
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm_bytes


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on punctuation."""
    import re
    parts = re.split(r"(?<=[.!?\n])\s*", text)
    return [p for p in parts if p]
