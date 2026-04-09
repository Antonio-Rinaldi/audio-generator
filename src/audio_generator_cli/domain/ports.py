from __future__ import annotations

from typing import Protocol

from audio_generator_cli.domain.models import AudioRequest, AudioResponse


class AudioGeneratorPort(Protocol):
    """Port for adapters that convert text to audio bytes."""

    def generate(self, request: AudioRequest, stream: bool = False) -> AudioResponse:
        """Generate audio for one text request."""
        raise NotImplementedError

