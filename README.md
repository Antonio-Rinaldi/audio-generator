# audio-generator-cli

`audio-generator` is a lightweight CLI that takes input text and generates one audio file using an OpenAI-compatible
`/v1/audio/speech` backend.

## Features

- Accept text directly (`--text`) or from file (`--text-file`).
- Split long input semantically before synthesis.
- Merge multi-chunk WAV responses into a single output file.
- Stream and non-stream request modes.
- OpenAI-compatible backend support (for local TTS servers).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Quick Start

```bash
audio-generate \
  --text "Ciao, questo e un test di sintesi vocale." \
  --out ./resources/output/output_1.wav \
  --voice-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --voice-base-url http://localhost:8000 \
  --voice nova_chunk_3
```

```bash
audio-generate \
  --text "Ciao, questo e un test di sintesi vocale." \
  --out ./resources/output/output_2.wav \
  --voice-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --voice-base-url http://localhost:8000 \
  --voice nova_chunk_3
```

## File Input Example

```bash
audio-generate \
  --text-file ./input.txt \
  --out ./output.wav \
  --voice-model mlx-community/Voxtral-4B-TTS-2603-mlx-4bit \
  --voice-base-url http://localhost:8000 \
  --stream
```

## CLI Options

- `--text`: text to synthesize.
- `--text-file`: path to file containing text.
- `--out`: output audio path (`.wav` by default).
- `--voice-model`: TTS model name.
- `--voice`: optional voice id.
- `--voice-base-url`: backend base URL (default `http://localhost:8000`).
- `--max-chars-per-request`: chunk size for each backend request.
- `--stream/--no-stream`: request streaming responses.
- `--log-level`: `INFO` or `DEBUG`.

## Test

```bash
pytest -q
```

