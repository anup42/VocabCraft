"""VocabCraft command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vocabcraft import __version__
from vocabcraft.config import load_config
from vocabcraft.exceptions import VocabCraftError
from vocabcraft.logging import configure_logging

app = typer.Typer(
    name="vocabcraft",
    help="Safe, profile-aware vocabulary compaction for multilingual models.",
    no_args_is_help=True,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"VocabCraft {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """Configure VocabCraft command execution."""

    del version
    configure_logging(verbose)


@app.command("check-config")
def check_config(
    profile: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a profile without downloading or loading a model."""

    try:
        config = load_config(profile)
    except VocabCraftError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"valid profile: {config.profile.id}")
