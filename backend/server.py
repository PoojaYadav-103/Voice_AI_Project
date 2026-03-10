"""
Real-Time Voice AI Backend
WebSocket gateway + STT + LLM + TTS pipeline
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol

from pipeline import VoicePipeline
from session import SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

session_manager = SessionManager()


async def handle_client(websocket: WebSocketServerProtocol, path: str):
    """Handle a new WebSocket client connection."""
    session_id = str(uuid.uuid4())
    logger.info(f"New client connected: {session_id}")

    pipeline = VoicePipeline(session_id=session_id, websocket=websocket)
    session_manager.add(session_id, pipeline)

    try:
        await websocket.send(json.dumps({"type": "session_started", "session_id": session_id}))

        async for message in websocket:
            if isinstance(message, bytes):
                # Raw audio chunk from browser
                await pipeline.handle_audio_chunk(message)
            elif isinstance(message, str):
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "interrupt":
                    await pipeline.handle_interrupt()
                elif msg_type == "end_of_speech":
                    await pipeline.handle_end_of_speech()
                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
    finally:
        await pipeline.cleanup()
        session_manager.remove(session_id)


async def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8765))

    logger.info(f"Starting Voice AI WebSocket server on ws://{host}:{port}")

    async with websockets.serve(
        handle_client,
        host,
        port,
        ping_interval=20,
        ping_timeout=10,
        max_size=1024 * 1024,  # 1MB max message
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
