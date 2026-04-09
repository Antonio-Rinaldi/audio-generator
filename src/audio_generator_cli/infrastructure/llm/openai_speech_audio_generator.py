from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass

import requests

from audio_generator_cli.domain.models import AudioRequest, AudioResponse
from audio_generator_cli.domain.ports import AudioGeneratorPort
from audio_generator_cli.infrastructure.logging.logger_factory import create_logger

logger = create_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class AudioGeneratorError(Exception):
    """Base error for audio generator failures."""


class RetryableAudioGeneratorError(AudioGeneratorError):
    """Retryable network/server error."""


class NonRetryableAudioGeneratorError(AudioGeneratorError):
    """Non-retryable input/backend contract error."""


def _response_error_excerpt(resp: requests.Response, max_len: int = 200) -> str:
    """Extract a short body excerpt from a failed response."""
    try:
        body = resp.content or b""
    except Exception:
        body = b""
    return body.decode("utf-8", errors="replace")[:max_len] if body else ""


@dataclass(frozen=True)
class SemanticTextChunker:
    """Split text into semantic chunks constrained by a max char budget."""

    max_chars: int

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        """Split text into non-empty paragraph blocks."""
        return [paragraph.strip() for paragraph in re.split(r"\n{2,}", text.strip()) if paragraph.strip()]

    @staticmethod
    def _sentences(paragraph: str) -> list[str]:
        """Split one paragraph into sentence-like units."""
        return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(paragraph) if sentence.strip()]

    def _slice_long_text(self, text: str) -> list[str]:
        """Slice a long text by max chars."""
        return [
            text[index : index + self.max_chars].strip()
            for index in range(0, len(text), self.max_chars)
            if text[index : index + self.max_chars].strip()
        ]

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        """Pack sentence list into max-sized semantic chunks."""
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(sentence) <= self.max_chars:
                current = sentence
            else:
                chunks.extend(self._slice_long_text(sentence))

        if current:
            chunks.append(current)
        return chunks

    def split(self, text: str) -> list[str]:
        """Return semantic chunks for backend calls."""
        paragraphs = self._paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                current = ""

            if len(paragraph) <= self.max_chars:
                current = paragraph
            else:
                sentences = self._sentences(paragraph)
                chunks.extend(self._pack_sentences(sentences) if len(sentences) > 1 else self._slice_long_text(paragraph))

        if current:
            chunks.append(current.strip())

        return chunks


def _concat_wav_bytes(parts: list[bytes]) -> bytes:
    """Concatenate WAV payloads preserving a single coherent header."""
    if not parts:
        return b""
    if len(parts) == 1:
        return parts[0]

    frames: list[bytes] = []
    params: tuple[int, int, int, str, str] | None = None

    for item in parts:
        with wave.open(io.BytesIO(item), "rb") as reader:
            current_params = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )
            if params is None:
                params = current_params
            elif params != current_params:
                raise RetryableAudioGeneratorError("Incompatible WAV chunks returned by TTS server")
            frames.append(reader.readframes(reader.getnframes()))

    if params is None:
        return b""

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(params[0])
        writer.setsampwidth(params[1])
        writer.setframerate(params[2])
        writer.setcomptype(params[3], params[4])
        for frame in frames:
            writer.writeframes(frame)
    return out.getvalue()


@dataclass(frozen=True)
class OpenAISpeechAudioGenerator(AudioGeneratorPort):
    """Audio generator using OpenAI-compatible ``/v1/audio/speech`` endpoint."""

    base_url: str = "http://localhost:8000"
    timeout_s: float = 600.0
    max_chars_per_request: int = 3900

    @staticmethod
    def _build_payload(request: AudioRequest, text_chunk: str) -> dict[str, str]:
        """Build backend payload for one chunk."""
        optional_payload = {key: value for key, value in (("voice", request.voice), ("instructions", request.instructions)) if value}
        return {
            "model": request.model,
            "input": text_chunk,
            "response_format": "wav",
            **optional_payload,
        }

    def _speech_url(self, stream: bool) -> str:
        """Build speech endpoint URL for stream or non-stream mode."""
        suffix = "?stream=true" if stream else ""
        return f"{self.base_url}/v1/audio/speech{suffix}"

    def _send_tts_request(self, payload: dict[str, str], stream: bool) -> requests.Response:
        """Send one TTS request and map transport failures."""
        try:
            if stream:
                return requests.post(self._speech_url(True), json=payload, timeout=self.timeout_s, stream=True)
            return requests.post(self._speech_url(False), json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise RetryableAudioGeneratorError(str(exc)) from exc

    @staticmethod
    def _extract_audio_bytes(resp: requests.Response, stream: bool) -> bytes:
        """Extract audio bytes from response content."""
        if stream:
            return b"".join(chunk for chunk in resp.iter_content(chunk_size=8192) if chunk)
        return resp.content

    @staticmethod
    def _detect_output_format(resp: requests.Response) -> str:
        """Infer output format from content type."""
        content_type = resp.headers.get("content-type", "audio/wav")
        return "mp3" if "mpeg" in content_type or "mp3" in content_type else "wav"

    @staticmethod
    def _validate_response(resp: requests.Response) -> None:
        """Validate backend response status codes."""
        if resp.status_code >= 500:
            raise RetryableAudioGeneratorError(f"TTS server error: {resp.status_code}")
        if resp.status_code >= 400:
            raise NonRetryableAudioGeneratorError(f"TTS request failed: {resp.status_code} {_response_error_excerpt(resp)}")

    def _fetch_chunk_audio(self, request: AudioRequest, text_chunk: str, stream: bool) -> tuple[bytes, str]:
        """Fetch one chunk of synthesized audio bytes."""
        response = self._send_tts_request(self._build_payload(request, text_chunk), stream=stream)
        self._validate_response(response)
        audio_bytes = self._extract_audio_bytes(response, stream=stream)
        if not audio_bytes:
            raise RetryableAudioGeneratorError("Empty audio response from TTS server")
        return audio_bytes, self._detect_output_format(response)

    @staticmethod
    def _merge_audio_chunks(audio_chunks: list[bytes], output_format: str) -> bytes:
        """Merge audio chunks according to final output format."""
        return _concat_wav_bytes(audio_chunks) if output_format == "wav" else b"".join(audio_chunks)

    def generate(self, request: AudioRequest, stream: bool = False) -> AudioResponse:
        """Generate one final audio response from input text request."""
        logger.debug(
            "Calling OpenAI-speech TTS | model=%s voice=%s text_len=%s stream=%s",
            request.model,
            request.voice or "(default)",
            len(request.text),
            stream,
        )

        text_chunks = SemanticTextChunker(self.max_chars_per_request).split(request.text)
        if not text_chunks:
            raise NonRetryableAudioGeneratorError("Empty text after preprocessing")

        chunk_results = [self._fetch_chunk_audio(request, text_chunk, stream) for text_chunk in text_chunks]
        audio_chunks = [audio for audio, _ in chunk_results]
        output_formats = {fmt for _, fmt in chunk_results}
        if len(output_formats) > 1:
            raise RetryableAudioGeneratorError("Inconsistent audio formats returned by TTS server")
        output_format = next(iter(output_formats), "wav")

        for index, (text_chunk, (audio_bytes, fmt)) in enumerate(zip(text_chunks, chunk_results), start=1):
            logger.debug(
                "OpenAI-speech TTS chunk received | chunk=%s/%s chars=%s bytes=%s fmt=%s stream=%s",
                index,
                len(text_chunks),
                len(text_chunk),
                len(audio_bytes),
                fmt,
                stream,
            )

        merged_audio = self._merge_audio_chunks(audio_chunks, output_format)
        logger.debug(
            "OpenAI-speech TTS response merged | model=%s chunks=%s bytes=%s fmt=%s stream=%s",
            request.model,
            len(text_chunks),
            len(merged_audio),
            output_format,
            stream,
        )
        return AudioResponse(audio_bytes=merged_audio, format=output_format)

