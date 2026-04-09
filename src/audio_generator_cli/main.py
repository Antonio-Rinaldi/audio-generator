from __future__ import annotations

import typer

from audio_generator_cli.cli import generate


def run() -> None:
    """Run Typer CLI entrypoint."""
    typer.run(generate)

