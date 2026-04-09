from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from audio_generator_cli.domain.models import AudioRequest
from audio_generator_cli.infrastructure.llm.openai_speech_audio_generator import (
    NonRetryableAudioGeneratorError,
    OpenAISpeechAudioGenerator,
    RetryableAudioGeneratorError,
)
from audio_generator_cli.infrastructure.logging.logger_factory import configure_logging, create_logger

console = Console()
logger = create_logger(__name__)

_OUTPUT_FORMATS = ("wav", "mp3")


@dataclass(frozen=True)
class GenerateTextCommand:
    """Validated CLI command payload used by text-to-audio generation."""

    text: str
    output_path: Path
    voice_model: str
    voice: str
    voice_base_url: str
    output_format: str
    log_level: str
    stream: bool
    max_chars_per_request: int
    instructions: str


def _abort(message: str) -> None:
    """Print error and terminate command with exit code 1."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


def _resolve_text(text: str, text_file: Path | None) -> str:
    """Resolve input text from direct argument or file path."""
    inline_text = text.strip()
    file_text = text_file.read_text(encoding="utf-8").strip() if text_file else ""

    if inline_text and file_text:
        _abort("Provide either --text or --text-file, not both.")
    if text_file and not text_file.exists():
        _abort(f"Text file not found: {text_file}")
    if text_file and not text_file.is_file():
        _abort(f"--text-file must point to a file: {text_file}")

    resolved_text = inline_text or file_text
    if not resolved_text:
        _abort("You must provide non-empty input with --text or --text-file.")
    return resolved_text


def _normalize_output_format(output_format: str) -> str:
    """Normalize output format and validate accepted values."""
    normalized = output_format.strip().lower()
    if normalized not in _OUTPUT_FORMATS:
        _abort(f"--output-format must be one of: {', '.join(_OUTPUT_FORMATS)}")
    return normalized


def _resolve_output_path(output_path: Path, output_format: str) -> Path:
    """Ensure output file has selected extension and parent directory exists."""
    resolved_path = output_path.with_suffix(f".{output_format}") if output_path.suffix.lower() != f".{output_format}" else output_path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _build_command(
    *,
    text: str,
    text_file: Path | None,
    output_path: Path,
    voice_model: str,
    voice: str,
    voice_base_url: str,
    output_format: str,
    log_level: str,
    stream: bool,
    max_chars_per_request: int,
    instructions: str,
) -> GenerateTextCommand:
    """Build immutable command payload after validation and normalization."""
    normalized_format = _normalize_output_format(output_format)
    resolved_text = _resolve_text(text=text, text_file=text_file)
    return GenerateTextCommand(
        text=resolved_text,
        output_path=_resolve_output_path(output_path, normalized_format),
        voice_model=voice_model,
        voice=voice,
        voice_base_url=voice_base_url or "http://localhost:8000",
        output_format=normalized_format,
        log_level=log_level,
        stream=stream,
        max_chars_per_request=max_chars_per_request,
        instructions=instructions.strip(),
    )


def _save_audio_file(path: Path, audio_bytes: bytes) -> None:
    """Write generated audio bytes to disk."""
    path.write_bytes(audio_bytes)


def generate(
    text: Annotated[str, typer.Option("--text", help="Text to synthesize") ] = "",
    text_file: Annotated[Optional[Path], typer.Option("--text-file", help="Path to input text file") ] = None,
    out_path: Annotated[Path, typer.Option("--out", help="Output audio path") ] = Path("output.wav"),
    voice_model: Annotated[str, typer.Option("--voice-model", help="TTS model name") ] = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    voice: Annotated[str, typer.Option("--voice", help="Voice id") ] = "alloy",
    voice_base_url: Annotated[str, typer.Option("--voice-base-url", help="TTS API base URL") ] = "http://localhost:8000",
    output_format: Annotated[str, typer.Option("--output-format", help="Output format: wav or mp3") ] = "wav",
    log_level: Annotated[str, typer.Option("--log-level", help="Logging level: DEBUG or INFO") ] = "INFO",
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="Use streamed backend response") ] = False,
    max_chars_per_request: Annotated[int, typer.Option("--max-chars-per-request", min=200, max=16000) ] = 3900,
    instructions: Annotated[str, typer.Option("--instructions", help="Optional speaking instructions") ] = "",
) -> None:
    """Convert input text to one audio output file."""
    command = _build_command(
        text=text,
        text_file=text_file,
        output_path=out_path,
        voice_model=voice_model,
        voice=voice,
        voice_base_url=voice_base_url,
        output_format=output_format,
        log_level=log_level,
        stream=stream,
        max_chars_per_request=max_chars_per_request,
        instructions=instructions,
    )
    configure_logging(command.log_level)

    logger.info(
        "Starting audio generation | model=%s voice=%s url=%s out=%s format=%s stream=%s chars=%s",
        command.voice_model,
        command.voice,
        command.voice_base_url,
        command.output_path,
        command.output_format,
        command.stream,
        len(command.text),
    )

    generator = OpenAISpeechAudioGenerator(
        base_url=command.voice_base_url,
        max_chars_per_request=command.max_chars_per_request,
    )

    try:
        response = generator.generate(
            AudioRequest(
                model=command.voice_model,
                text=command.text,
                voice=command.voice,
                instructions=command.instructions,
            ),
            stream=command.stream,
        )
    except NonRetryableAudioGeneratorError as exc:
        _abort(str(exc))
    except RetryableAudioGeneratorError as exc:
        _abort(f"Temporary backend error: {exc}")

    _save_audio_file(command.output_path, response.audio_bytes)
    logger.info("Audio generation finished | path=%s bytes=%s", command.output_path, len(response.audio_bytes))

    console.print(
        json.dumps(
            {
                "out_path": str(command.output_path),
                "audio_format": response.format,
                "bytes": len(response.audio_bytes),
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0)

