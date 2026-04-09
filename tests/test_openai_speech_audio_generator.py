from __future__ import annotations

import io
import wave

from audio_generator_cli.domain.models import AudioRequest
from audio_generator_cli.infrastructure.llm.openai_speech_audio_generator import (
    OpenAISpeechAudioGenerator,
    SemanticTextChunker,
)


class _FakeResponse:
    def __init__(self, audio: bytes) -> None:
        self.status_code = 200
        self.content = audio
        self.headers = {"content-type": "audio/wav"}

    def iter_content(self, chunk_size: int = 8192):
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]


def _wav_bytes(frames: int, framerate: int = 22050) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(framerate)
        writer.writeframes(b"\x00\x00" * frames)
    return out.getvalue()


def test_semantic_text_chunker_respects_limit() -> None:
    chunker = SemanticTextChunker(max_chars=20)
    chunks = chunker.split("Prima frase lunga. Seconda frase lunga. Terza frase lunga.")

    assert chunks
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_generate_splits_and_merges_wav(monkeypatch) -> None:
    calls: list[str] = []
    first = _wav_bytes(100)
    second = _wav_bytes(200)

    def _fake_post(url: str, json: dict, timeout: float):
        _ = url
        _ = timeout
        calls.append(json["input"])
        return _FakeResponse(first if len(calls) == 1 else second)

    monkeypatch.setattr(
        "audio_generator_cli.infrastructure.llm.openai_speech_audio_generator.requests.post",
        _fake_post,
    )

    generator = OpenAISpeechAudioGenerator(max_chars_per_request=22)
    response = generator.generate(
        AudioRequest(
            model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            text="Prima frase molto lunga. Seconda frase ancora lunga. Terza frase finale.",
            voice="gold",
        )
    )

    assert response.format == "wav"
    assert len(calls) >= 2

    expected_frames = 100 + 200 * (len(calls) - 1)
    with wave.open(io.BytesIO(response.audio_bytes), "rb") as reader:
        assert reader.getnframes() == expected_frames

