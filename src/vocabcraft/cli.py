"""VocabCraft command-line interface."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn, TypeVar

import typer

from vocabcraft import __version__
from vocabcraft.config import load_config
from vocabcraft.exceptions import VocabCraftError
from vocabcraft.inspection import inspect_mt5_to_directory
from vocabcraft.logging import configure_logging
from vocabcraft.workflows import (
    ExecutionMode,
    analyze_vocab_to_directory,
    benchmark_to_directory,
    build_pack_to_directory,
    build_profile_to_directory,
    merge_packs_to_directory,
    reconstruct_full_to_directory,
    validate_to_directory,
)

app = typer.Typer(
    name="vocabcraft",
    help="Safe, profile-aware vocabulary compaction for multilingual models.",
    no_args_is_help=True,
)

T = TypeVar("T")


class Mode(StrEnum):
    """User-facing profile execution modes."""

    encoder_exact = "encoder_exact"
    seq2seq_compact = "seq2seq_compact"
    seq2seq_guarded = "seq2seq_guarded"


def _abort(exc: Exception) -> NoReturn:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=2) from exc


def _run(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except (VocabCraftError, OSError, ValueError) as exc:
        _abort(exc)


def _print_result(output: Path, result: dict[str, Any]) -> None:
    typer.echo(
        json.dumps(
            {"output": str(output.resolve()), "result": result},
            ensure_ascii=False,
            sort_keys=True,
        )
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

    config = _run(lambda: load_config(profile))
    typer.echo(f"valid profile: {config.profile.id}")


@app.command("inspect-model")
def inspect_model_command(
    model: Annotated[str, typer.Option(help="Model identifier or local checkpoint.")],
    output: Annotated[Path, typer.Option(help="New output directory for inspection reports.")],
    revision: Annotated[str | None, typer.Option(help="Optional exact model revision.")] = None,
) -> None:
    """Inspect actual mT5 tensors, IDs, storage ties, and tokenizer metadata."""

    result = _run(
        lambda: inspect_mt5_to_directory(model, output, revision=revision, trust_remote_code=False)
    )
    _print_result(output, result)


@app.command("analyze-vocab")
def analyze_vocab_command(
    model: Annotated[str, typer.Option(help="Model identifier or local checkpoint.")],
    profile: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True, help="Profile YAML.")
    ],
    output: Annotated[Path, typer.Option(help="New output directory for manifests.")],
) -> None:
    """Build an inventory and conservative retained/excluded manifests."""

    config = _run(lambda: load_config(profile))
    result = _run(lambda: analyze_vocab_to_directory(model, config, output))
    _print_result(output, result)


@app.command("build-profile")
def build_profile_command(
    model: Annotated[str, typer.Option(help="Model identifier or local checkpoint.")],
    profile: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True, help="Profile YAML.")
    ],
    mode: Annotated[Mode, typer.Option(help="Compaction execution mode.")],
    output: Annotated[Path, typer.Option(help="New compact artifact directory.")],
) -> None:
    """Build an exact encoder or explicitly experimental seq2seq profile."""

    config = _run(lambda: load_config(profile))
    execution_mode: ExecutionMode = mode.value
    result = _run(lambda: build_profile_to_directory(model, config, execution_mode, output))
    _print_result(output, result)


@app.command("validate")
def validate_command(
    original: Annotated[str, typer.Option(help="Original model identifier/checkpoint.")],
    compact: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True, help="Compact artifact.")
    ],
    profile: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True, help="Profile YAML.")
    ],
    data: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True, help="JSONL data.")
    ],
    output: Annotated[Path, typer.Option(help="New validation report directory.")],
) -> None:
    """Run coverage and mode-specific before-versus-after validation."""

    config = _run(lambda: load_config(profile))
    result = _run(lambda: validate_to_directory(original, compact, config, data, output))
    _print_result(output, result)
    if not bool(result["passed"]):
        raise typer.Exit(code=3)


@app.command("benchmark")
def benchmark_command(
    original: Annotated[str, typer.Option(help="Original model identifier/checkpoint.")],
    compact: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True, help="Compact artifact.")
    ],
    data: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True, help="JSONL data.")
    ],
    output: Annotated[Path, typer.Option(help="New benchmark report directory.")],
) -> None:
    """Measure CPU sizes, memory, mapping overhead, and encoder latency."""

    result = _run(lambda: benchmark_to_directory(original, compact, data, output))
    _print_result(output, result)


@app.command("build-pack")
def build_pack_command(
    model: Annotated[str, typer.Option(help="Original model identifier/checkpoint.")],
    compact: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True, help="Compact artifact.")
    ],
    output: Annotated[Path, typer.Option(help="New cold pack directory.")],
) -> None:
    """Export exact excluded vocabulary rows as a verified cold pack."""

    result = _run(lambda: build_pack_to_directory(model, compact, output))
    _print_result(output, result)


@app.command("merge-packs")
def merge_packs_command(
    packs: Annotated[
        list[Path], typer.Option("--pack", exists=True, file_okay=False, help="Input pack.")
    ],
    output: Annotated[Path, typer.Option(help="New merged pack directory.")],
) -> None:
    """Merge compatible packs deterministically; conflicting rows fail."""

    result = _run(lambda: merge_packs_to_directory(list(packs), output))
    _print_result(output, result)


@app.command("reconstruct-full")
def reconstruct_full_command(
    compact: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True, help="Compact artifact.")
    ],
    pack: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True, help="Cold/merged pack.")
    ],
    output: Annotated[Path, typer.Option(help="New reconstructed artifact directory.")],
) -> None:
    """Reconstruct and reload the original-vocabulary mT5 tensors offline."""

    result = _run(lambda: reconstruct_full_to_directory(compact, pack, output))
    _print_result(output, result)
