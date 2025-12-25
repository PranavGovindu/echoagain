"""
Custom Pipecat TTS Service for Echo TTS.

This service connects to an Echo TTS server via HTTP streaming
and streams audio back to the Pipecat pipeline.
"""

import asyncio
import os
from typing import AsyncGenerator, Optional

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
        
        logger.info(f"Echo TTS Service initialized with server: {self._server_url}")

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

        # Build HTTP endpoint URL
        endpoint_url = f"{self._server_url}/v1/audio/speech"
        logger.debug(f"Connecting to Echo TTS HTTP endpoint: {endpoint_url}")

        # Prepare request payload matching Echo TTS API format
        payload = {
            "model": "echo-tts",  # Optional but included for compatibility
            "input": text,
            "voice": self._voice,
            "response_format": "pcm",  # Raw PCM for streaming
            "stream": True,
            "extra_body": {
                "cfg_scale_text": self._cfg_scale_text,
                "cfg_scale_speaker": self._cfg_scale_speaker,
                "seed": self._seed,
            }
        }

        try:
            # Signal TTS started
            yield TTSStartedFrame()
            
            # Make HTTP POST request with streaming
            async with aiohttp.ClientSession() as session:
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
                        raise Exception(error_msg)
                    
                    # Check content type header
                    content_type = response.headers.get("Content-Type", "")
                    sample_rate_header = response.headers.get("X-Audio-Sample-Rate")
                    if sample_rate_header:
                        try:
                            detected_rate = int(sample_rate_header)
                            if detected_rate != self._sample_rate:
                                logger.warning(
                                    f"Server sample rate ({detected_rate}) differs from configured ({self._sample_rate}). "
                                    f"Using server rate."
                                )
                                self._sample_rate = detected_rate
                        except ValueError:
                            pass
                    
                    logger.debug(f"Streaming response: Content-Type={content_type}, Sample-Rate={sample_rate_header}")
                    
                    # Buffer for incomplete PCM16 frames (2 bytes per sample)
                    # PCM16 requires chunks to be multiples of 2 bytes
                    buffer = bytearray()
                    bytes_per_sample = 2  # int16 = 2 bytes
                    
                    # Stream audio chunks as they arrive
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            buffer.extend(chunk)
                            
                            # Only yield complete frames (multiples of 2 bytes)
                            complete_samples = len(buffer) // bytes_per_sample
                            if complete_samples > 0:
                                complete_bytes = complete_samples * bytes_per_sample
                                frame_data = bytes(buffer[:complete_bytes])
                                buffer = buffer[complete_bytes:]
                                
                                # Echo TTS sends PCM16 (int16) bytes at 44.1kHz
                                yield TTSAudioRawFrame(
                                    audio=frame_data,
                                    sample_rate=self._sample_rate,
                                    num_channels=1,
                                )
                    
                    # Yield any remaining buffered data (should be empty, but handle it)
                    if len(buffer) > 0:
                        # Pad to complete sample if needed (shouldn't happen, but safety)
                        if len(buffer) % bytes_per_sample != 0:
                            buffer.extend(b'\x00' * (bytes_per_sample - (len(buffer) % bytes_per_sample)))
                        if len(buffer) > 0:
                            yield TTSAudioRawFrame(
                                audio=bytes(buffer),
                                sample_rate=self._sample_rate,
                                num_channels=1,
                            )
            
            # Signal TTS stopped
            yield TTSStoppedFrame()
                
        except aiohttp.ClientError as e:
            logger.error(f"Echo TTS connection error: {e} | URL: {endpoint_url}")
            yield TTSStoppedFrame()
        except Exception as e:
            logger.error(f"Echo TTS error: {e} | URL: {endpoint_url}")
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

