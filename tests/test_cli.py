from __future__ import annotations

from pathlib import Path

import pytest
import typer

from audio_generator_cli import cli


def test_resolve_text_from_inline() -> None:
    assert cli._resolve_text("ciao", None) == "ciao"


def test_resolve_text_rejects_both_inline_and_file(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("contenuto", encoding="utf-8")

    with pytest.raises(typer.Exit):
        cli._resolve_text("ciao", text_file)


def test_normalize_output_format_rejects_invalid() -> None:
    with pytest.raises(typer.Exit):
        cli._normalize_output_format("ogg")


