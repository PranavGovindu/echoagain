"""
Custom Pipecat TTS Service for Echo TTS.

This service connects to an Echo TTS server via HTTP streaming
and streams audio back to the Pipecat pipeline.
"""

import asyncio
import os
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    raise ImportError("aiohttp is required. Install with: pip install aiohttp")

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from loguru import logger


class EchoTTSService(TTSService):
    """
    Echo TTS Service for Pipecat.
    
    Connects to an Echo TTS server via HTTP streaming and streams
    PCM16 audio chunks (44.1kHz, int16, mono).
    """

    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        voice: str = "expresso_02_ex03-ex01_calm_005",
        cfg_scale_text: float = 2.5,
        cfg_scale_speaker: float = 5.0,
        seed: int = 0,
        sample_rate: int = 44100,
        transport: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Echo TTS Service.
        
        Args:
            server_url: HTTP URL of Echo TTS server (e.g., "http://localhost:8000")
                       If not provided, uses ECHO_SERVER_URL env var.
            voice: Voice name from audio_prompts/ directory
            cfg_scale_text: Text classifier-free guidance scale (default: 2.5)
            cfg_scale_speaker: Speaker classifier-free guidance scale (default: 5.0)
            seed: Random seed for reproducibility (default: 0)
            sample_rate: Audio sample rate in Hz (default: 44100)
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._server_url = server_url or os.environ.get("ECHO_SERVER_URL")
        if not self._server_url:
            raise ValueError(
                "Echo TTS server URL is required. "
                "Provide server_url parameter or set ECHO_SERVER_URL environment variable."
            )
        
        # Normalize URL - ensure it's HTTP/HTTPS
        if self._server_url.startswith("ws://"):
            self._server_url = "http://" + self._server_url[5:]
        elif self._server_url.startswith("wss://"):
            self._server_url = "https://" + self._server_url[6:]
        elif not self._server_url.startswith(("http://", "https://")):
            self._server_url = "http://" + self._server_url
        
        # Remove trailing slash if present
        self._server_url = self._server_url.rstrip("/")
        
        self._voice = voice
        self._cfg_scale_text = cfg_scale_text
        self._cfg_scale_speaker = cfg_scale_speaker
        self._seed = seed
        self._sample_rate = sample_rate
        
        transport_raw = transport or os.environ.get("ECHO_TTS_TRANSPORT", "http")
        self._transport = transport_raw.strip().lower()
        if self._transport not in {"http", "ws", "auto"}:
            logger.warning("Unknown ECHO_TTS_TRANSPORT '{}', defaulting to http", self._transport)
            self._transport = "http"

        logger.info(
            f"Echo TTS Service initialized with server: {self._server_url} (transport={self._transport})"
        )

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, value: str):
        self._voice = value
        logger.info(f"Echo TTS voice changed to: {value}")

    async def set_voice(self, voice: str):
        """Set the voice preset."""
        self._voice = voice
        logger.info(f"Echo TTS voice set to: {voice}")

    async def start(self, frame: StartFrame):
        """Called when the pipeline starts."""
        await super().start(frame)
        logger.info("Echo TTS Service started")

    async def stop(self, frame: EndFrame):
        """Called when the pipeline stops."""
        await super().stop(frame)
        logger.info("Echo TTS Service stopped")

    async def cancel(self, frame: CancelFrame):
        """Called when generation is cancelled."""
        await super().cancel(frame)
        logger.info("Echo TTS generation cancelled")

    def _build_websocket_url(self, text: str) -> str:
        base_url = self._server_url
        if base_url.startswith("http://"):
            base_url = "ws://" + base_url[7:]
        elif base_url.startswith("https://"):
            base_url = "wss://" + base_url[8:]
        elif not base_url.startswith(("ws://", "wss://")):
            base_url = "ws://" + base_url

        params = {
            "text": text,
            "voice": self._voice,
            "cfg_scale_text": str(self._cfg_scale_text),
            "cfg_scale_speaker": str(self._cfg_scale_speaker),
            "seed": str(self._seed),
        }
        return f"{base_url}/stream?{urlencode(params)}"

    async def _run_tts_http(self, text: str) -> AsyncGenerator[Frame, None]:
        endpoint_url = f"{self._server_url}/v1/audio/speech"
        logger.debug(f"Connecting to Echo TTS HTTP endpoint: {endpoint_url}")

        payload = {
            "model": "echo-tts",
            "input": text,
            "voice": self._voice,
            "response_format": "pcm",
            "stream": True,
            "extra_body": {
                "cfg_scale_text": self._cfg_scale_text,
                "cfg_scale_speaker": self._cfg_scale_speaker,
                "seed": self._seed,
            },
        }

        yield TTSStartedFrame()

        timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                endpoint_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/octet-stream",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"Echo TTS HTTP error {response.status}: {error_text}"
                    logger.error(f"{error_msg} | URL: {endpoint_url} | Payload keys: {list(payload.keys())}")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_text,
                        headers=response.headers,
                    )

                content_type = response.headers.get("Content-Type", "")
                sample_rate_header = response.headers.get("X-Audio-Sample-Rate")
                if sample_rate_header:
                    try:
                        detected_rate = int(sample_rate_header)
                        if detected_rate != self._sample_rate:
                            logger.warning(
                                "Server sample rate ({}) differs from configured ({}). Using server rate.",
                                detected_rate,
                                self._sample_rate,
                            )
                            self._sample_rate = detected_rate
                    except ValueError:
                        pass

                logger.debug(
                    "Streaming response: Content-Type={}, Sample-Rate={}",
                    content_type,
                    sample_rate_header,
                )

                buffer = bytearray()
                bytes_per_sample = 2
                try:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            buffer.extend(chunk)
                            complete_samples = len(buffer) // bytes_per_sample
                            if complete_samples > 0:
                                complete_bytes = complete_samples * bytes_per_sample
                                frame_data = bytes(buffer[:complete_bytes])
                                buffer = buffer[complete_bytes:]
                                yield TTSAudioRawFrame(
                                    audio=frame_data,
                                    sample_rate=self._sample_rate,
                                    num_channels=1,
                                )
                except aiohttp.ClientPayloadError as exc:
                    logger.warning("Echo TTS stream ended early: {}", exc)

                if len(buffer) > 0:
                    if len(buffer) % bytes_per_sample != 0:
                        buffer.extend(b"\x00" * (bytes_per_sample - (len(buffer) % bytes_per_sample)))
                    if len(buffer) > 0:
                        yield TTSAudioRawFrame(
                            audio=bytes(buffer),
                            sample_rate=self._sample_rate,
                            num_channels=1,
                        )

        yield TTSStoppedFrame()

    async def _run_tts_ws(self, text: str) -> AsyncGenerator[Frame, None]:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError("websockets is required for ws transport. Install with: pip install websockets") from exc

        ws_url = self._build_websocket_url(text)
        logger.debug(f"Connecting to Echo TTS WebSocket: {ws_url[:100]}...")

        yield TTSStartedFrame()
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=None,
        ) as websocket:
            async for message in websocket:
                if isinstance(message, bytes):
                    yield TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                elif isinstance(message, str):
                    logger.debug(f"Echo TTS log: {message}")

        yield TTSStoppedFrame()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech from text using Echo TTS server.
        
        Args:
            text: Text to convert to speech.
            
        Yields:
            TTSAudioRawFrame: Audio frames containing PCM audio data.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to Echo TTS")
            return

        logger.debug(f"Echo TTS generating speech for: {text[:50]}...")

        try:
            if self._transport == "ws":
                async for frame in self._run_tts_ws(text):
                    yield frame
            elif self._transport == "auto":
                try:
                    async for frame in self._run_tts_http(text):
                        yield frame
                except aiohttp.ClientError as exc:
                    logger.warning("HTTP TTS failed, retrying over WebSocket: {}", exc)
                    async for frame in self._run_tts_ws(text):
                        yield frame
            else:
                async for frame in self._run_tts_http(text):
                    yield frame
        except aiohttp.ClientError as e:
            logger.error("Echo TTS connection error: {}", e)
            yield TTSStoppedFrame()
        except Exception as e:
            logger.error("Echo TTS error: {}", e)
            yield TTSStoppedFrame()

    async def get_available_voices(self) -> list[str]:
        """
        Fetch available voices from Echo TTS server.
        
        Returns:
            List of available voice names.
        """
        voices_url = f"{self._server_url}/v1/voices"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(voices_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Echo TTS returns {"object": "list", "data": [{"id": "voice_name", ...}]}
                        return [v.get("id", v.get("name", "")) for v in data.get("data", [])]
                    else:
                        logger.warning(f"Failed to fetch voices: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching voices: {e}")
            return []
