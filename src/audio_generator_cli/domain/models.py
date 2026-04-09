from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioRequest:
    """Text-to-speech request payload for audio generator adapters."""

    model: str
    text: str
    voice: str = ""
    instructions: str = ""


@dataclass(frozen=True)
class AudioResponse:
    """Raw audio payload returned by an audio generator adapter."""

    audio_bytes: bytes
    format: str = "wav"

