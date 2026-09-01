"""Validated YAML configuration for VocabCraft profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from vocabcraft.exceptions import ConfigurationError


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must be a list of strings")
    return cast(list[str], value)


def _int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ConfigurationError(f"{name} must be a list of integers")
    return cast(list[int], value)


@dataclass(frozen=True)
class ProjectConfig:
    """Project identity recorded in generated artifacts."""

    name: str = "VocabCraft"
    version: str = "0.1"


@dataclass(frozen=True)
class ModelConfig:
    """Source model loading configuration."""

    id: str
    revision: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ProfileConfig:
    """Language, domain, and retention rules for a compact profile."""

    id: str
    description: str
    languages: list[str]
    unicode_scripts: list[str]
    allow_common_script: bool = True
    allow_inherited_script: bool = True
    allow_code_switching: bool = True
    allow_romanized_hindi: bool = True
    allow_ascii_digits: bool = True
    allow_native_digits: bool = True
    allow_punctuation: bool = True
    allow_currency_symbols: bool = True
    allow_math_symbols: bool = True
    allow_urls: bool = True
    allow_emails: bool = True
    allow_emoji: bool = True
    keep_all_special_tokens: bool = True
    keep_all_control_tokens: bool = True
    keep_all_extra_ids: bool = True
    keep_unknown_token: bool = True
    keep_single_character_pieces_for_allowed_scripts: bool = True
    calibration_files: list[Path] = field(default_factory=list)
    critical_terms_files: list[Path] = field(default_factory=list)
    manual_keep_ids: list[int] = field(default_factory=list)
    manual_remove_ids: list[int] = field(default_factory=list)
    manual_keep_pieces: list[str] = field(default_factory=list)
    manual_remove_pieces: list[str] = field(default_factory=list)
    application_reserved_tokens: list[str] = field(default_factory=list)
    strict: bool = True


FallbackPolicy = Literal["full_model", "reject", "callback"]


@dataclass(frozen=True)
class FallbackConfig:
    """Fail-closed runtime behavior for missing token IDs."""

    policy: FallbackPolicy = "full_model"
    fail_closed_when_unavailable: bool = True
    record_missing_pieces: bool = True


@dataclass(frozen=True)
class ValidationConfig:
    """Configurable validation tolerances; defaults are not universal thresholds."""

    maximum_new_unk_count: int = 0
    maximum_missing_critical_examples: int = 0
    encoder_max_absolute_difference: float = 1.0e-5
    encoder_mean_absolute_difference: float = 1.0e-6
    minimum_encoder_cosine_similarity: float = 0.999999
    teacher_forcing_max_absolute_difference: float = 5.0e-5
    generation_exact_match_required: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution device and batching configuration."""

    dtype: str = "auto"
    device: str = "cpu"
    batch_size: int = 4


@dataclass(frozen=True)
class NonInferiorityConfig:
    """Metric and maximum allowed aggregate drop for an external evaluation."""

    metric: str
    maximum_allowed_drop: float


@dataclass(frozen=True)
class ExternalEvaluationConfig:
    """Local-only hidden evaluation command and declared metrics output."""

    command: list[str]
    metrics_file: Path
    timeout_seconds: int
    non_inferiority: NonInferiorityConfig


@dataclass(frozen=True)
class VocabCraftConfig:
    """Fully validated VocabCraft configuration."""

    project: ProjectConfig
    model: ModelConfig
    profile: ProfileConfig
    fallback: FallbackConfig
    validation: ValidationConfig
    runtime: RuntimeConfig
    external_evaluation: ExternalEvaluationConfig | None
    source_path: Path


def _paths(values: object, name: str, base: Path) -> list[Path]:
    paths = _string_list(values, name)
    return [path if (path := Path(value)).is_absolute() else base / path for value in paths]


def load_config(path: str | Path) -> VocabCraftConfig:
    """Load and validate a YAML profile without initiating model downloads."""

    source = Path(path).resolve()
    try:
        raw_object = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration {source}: {exc}") from exc
    raw = _mapping(raw_object, "configuration")
    project_raw = _mapping(raw.get("project", {}), "project")
    model_raw = _mapping(raw.get("model"), "model")
    profile_raw = _mapping(raw.get("profile"), "profile")
    fallback_raw = _mapping(raw.get("fallback", {}), "fallback")
    validation_raw = _mapping(raw.get("validation", {}), "validation")
    runtime_raw = _mapping(raw.get("runtime", {}), "runtime")
    external_raw_object = raw.get("external_evaluation")

    model_id = model_raw.get("id")
    profile_id = profile_raw.get("id")
    description = profile_raw.get("description")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigurationError("model.id must be a non-empty string")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ConfigurationError("profile.id must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ConfigurationError("profile.description must be a non-empty string")

    policy = fallback_raw.get("policy", "full_model")
    if policy not in {"full_model", "reject", "callback"}:
        raise ConfigurationError("fallback.policy must be full_model, reject, or callback")
    manual_keep_ids = _int_list(profile_raw.get("manual_keep_ids", []), "manual_keep_ids")
    manual_remove_ids = _int_list(profile_raw.get("manual_remove_ids", []), "manual_remove_ids")
    overlap = sorted(set(manual_keep_ids) & set(manual_remove_ids))
    if overlap:
        raise ConfigurationError(f"IDs cannot be both manually kept and removed: {overlap}")

    batch_size = runtime_raw.get("batch_size", 4)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigurationError("runtime.batch_size must be a positive integer")
    device = runtime_raw.get("device", "cpu")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("runtime.device must be a non-empty string")
    revision = model_raw.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ConfigurationError("model.revision must be null or a string")

    external_evaluation: ExternalEvaluationConfig | None = None
    if external_raw_object is not None:
        external_raw = _mapping(external_raw_object, "external_evaluation")
        non_inferiority_raw = _mapping(
            external_raw.get("non_inferiority"), "external_evaluation.non_inferiority"
        )
        command = _string_list(external_raw.get("command"), "external_evaluation.command")
        if not command:
            raise ConfigurationError("external_evaluation.command cannot be empty")
        metrics_file = external_raw.get("metrics_file")
        metric = non_inferiority_raw.get("metric")
        timeout = external_raw.get("timeout_seconds", 3600)
        maximum_drop = non_inferiority_raw.get("maximum_allowed_drop")
        if not isinstance(metrics_file, str) or not metrics_file:
            raise ConfigurationError("external_evaluation.metrics_file must be a path string")
        if not isinstance(metric, str) or not metric:
            raise ConfigurationError("external_evaluation.non_inferiority.metric is required")
        if not isinstance(timeout, int) or timeout < 1:
            raise ConfigurationError("external_evaluation.timeout_seconds must be positive")
        if not isinstance(maximum_drop, (int, float)) or maximum_drop < 0:
            raise ConfigurationError(
                "external_evaluation.non_inferiority.maximum_allowed_drop must be non-negative"
            )
        metrics_path = Path(metrics_file)
        external_evaluation = ExternalEvaluationConfig(
            command=command,
            metrics_file=(
                metrics_path if metrics_path.is_absolute() else source.parent / metrics_path
            ),
            timeout_seconds=timeout,
            non_inferiority=NonInferiorityConfig(metric, float(maximum_drop)),
        )

    config = VocabCraftConfig(
        project=ProjectConfig(
            name=str(project_raw.get("name", "VocabCraft")),
            version=str(project_raw.get("version", "0.1")),
        ),
        model=ModelConfig(
            id=model_id,
            revision=revision,
            trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
        ),
        profile=ProfileConfig(
            id=profile_id,
            description=description,
            languages=_string_list(profile_raw.get("languages", []), "profile.languages"),
            unicode_scripts=_string_list(
                profile_raw.get("unicode_scripts", []), "profile.unicode_scripts"
            ),
            allow_common_script=bool(profile_raw.get("allow_common_script", True)),
            allow_inherited_script=bool(profile_raw.get("allow_inherited_script", True)),
            allow_code_switching=bool(profile_raw.get("allow_code_switching", True)),
            allow_romanized_hindi=bool(profile_raw.get("allow_romanized_hindi", True)),
            allow_ascii_digits=bool(profile_raw.get("allow_ascii_digits", True)),
            allow_native_digits=bool(profile_raw.get("allow_native_digits", True)),
            allow_punctuation=bool(profile_raw.get("allow_punctuation", True)),
            allow_currency_symbols=bool(profile_raw.get("allow_currency_symbols", True)),
            allow_math_symbols=bool(profile_raw.get("allow_math_symbols", True)),
            allow_urls=bool(profile_raw.get("allow_urls", True)),
            allow_emails=bool(profile_raw.get("allow_emails", True)),
            allow_emoji=bool(profile_raw.get("allow_emoji", True)),
            keep_all_special_tokens=bool(profile_raw.get("keep_all_special_tokens", True)),
            keep_all_control_tokens=bool(profile_raw.get("keep_all_control_tokens", True)),
            keep_all_extra_ids=bool(profile_raw.get("keep_all_extra_ids", True)),
            keep_unknown_token=bool(profile_raw.get("keep_unknown_token", True)),
            keep_single_character_pieces_for_allowed_scripts=bool(
                profile_raw.get("keep_single_character_pieces_for_allowed_scripts", True)
            ),
            calibration_files=_paths(
                profile_raw.get("calibration_files", []), "profile.calibration_files", source.parent
            ),
            critical_terms_files=_paths(
                profile_raw.get("critical_terms_files", []),
                "profile.critical_terms_files",
                source.parent,
            ),
            manual_keep_ids=manual_keep_ids,
            manual_remove_ids=manual_remove_ids,
            manual_keep_pieces=_string_list(
                profile_raw.get("manual_keep_pieces", []), "manual_keep_pieces"
            ),
            manual_remove_pieces=_string_list(
                profile_raw.get("manual_remove_pieces", []), "manual_remove_pieces"
            ),
            application_reserved_tokens=_string_list(
                profile_raw.get("application_reserved_tokens", []),
                "application_reserved_tokens",
            ),
            strict=bool(profile_raw.get("strict", True)),
        ),
        fallback=FallbackConfig(
            policy=cast(FallbackPolicy, policy),
            fail_closed_when_unavailable=bool(
                fallback_raw.get("fail_closed_when_unavailable", True)
            ),
            record_missing_pieces=bool(fallback_raw.get("record_missing_pieces", True)),
        ),
        validation=ValidationConfig(
            maximum_new_unk_count=int(validation_raw.get("maximum_new_unk_count", 0)),
            maximum_missing_critical_examples=int(
                validation_raw.get("maximum_missing_critical_examples", 0)
            ),
            encoder_max_absolute_difference=float(
                validation_raw.get("encoder_max_absolute_difference", 1.0e-5)
            ),
            encoder_mean_absolute_difference=float(
                validation_raw.get("encoder_mean_absolute_difference", 1.0e-6)
            ),
            minimum_encoder_cosine_similarity=float(
                validation_raw.get("minimum_encoder_cosine_similarity", 0.999999)
            ),
            teacher_forcing_max_absolute_difference=float(
                validation_raw.get("teacher_forcing_max_absolute_difference", 5.0e-5)
            ),
            generation_exact_match_required=bool(
                validation_raw.get("generation_exact_match_required", False)
            ),
        ),
        runtime=RuntimeConfig(
            dtype=str(runtime_raw.get("dtype", "auto")),
            device=device,
            batch_size=batch_size,
        ),
        external_evaluation=external_evaluation,
        source_path=source,
    )
    if config.project.name != "VocabCraft":
        raise ConfigurationError("project.name must be VocabCraft")
    if (
        config.validation.encoder_max_absolute_difference < 0
        or config.validation.encoder_mean_absolute_difference < 0
        or config.validation.teacher_forcing_max_absolute_difference < 0
    ):
        raise ConfigurationError("validation difference tolerances must be non-negative")
    if not -1.0 <= config.validation.minimum_encoder_cosine_similarity <= 1.0:
        raise ConfigurationError(
            "validation.minimum_encoder_cosine_similarity must be between -1 and 1"
        )
    return config
