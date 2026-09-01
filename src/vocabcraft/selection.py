"""Conservative, reason-recording profile token selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vocabcraft.config import ProfileConfig
from vocabcraft.evaluation.datasets import CalibrationRecord, stream_jsonl
from vocabcraft.exceptions import ConfigurationError, ValidationFailure
from vocabcraft.inventory import TokenRecord
from vocabcraft.tokenizers.base import OriginalTokenizer
from vocabcraft.unicode_analysis import is_url_email_building_piece, visible_piece

_CONFIG_TOKEN_FIELDS = (
    "pad_token_id",
    "eos_token_id",
    "bos_token_id",
    "decoder_start_token_id",
    "unk_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
)
_GENERATION_LIST_FIELDS = ("suppress_tokens", "begin_suppress_tokens")


@dataclass(frozen=True)
class SelectionResult:
    """Deterministic retained/excluded IDs and their annotated inventory."""

    retained_ids: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    records: tuple[TokenRecord, ...]


def _valid_id(value: object, vocabulary_size: int) -> bool:
    return isinstance(value, int) and 0 <= value < vocabulary_size


def mandatory_token_ids(
    tokenizer: OriginalTokenizer,
    model_config: Any | None = None,
    generation_config: Any | None = None,
    application_reserved_tokens: Iterable[str] = (),
) -> set[int]:
    """Collect every token ID explicitly required by tokenizer/model configuration."""

    vocabulary_size = len(tokenizer)
    mandatory = {
        token_id for token_id in tokenizer.all_special_ids if _valid_id(token_id, vocabulary_size)
    }
    for owner in (model_config, generation_config):
        if owner is None:
            continue
        for name in _CONFIG_TOKEN_FIELDS:
            value = getattr(owner, name, None)
            if isinstance(value, int) and _valid_id(value, vocabulary_size):
                mandatory.add(value)
        for name in _GENERATION_LIST_FIELDS:
            value = getattr(owner, name, None)
            if isinstance(value, list):
                mandatory.update(item for item in value if _valid_id(item, vocabulary_size))
        bad_words = getattr(owner, "bad_words_ids", None)
        if isinstance(bad_words, list):
            for sequence in bad_words:
                if isinstance(sequence, list):
                    mandatory.update(item for item in sequence if _valid_id(item, vocabulary_size))
    for token in application_reserved_tokens:
        value = tokenizer.convert_tokens_to_ids(token)
        if isinstance(value, int) and _valid_id(value, vocabulary_size):
            mandatory.add(value)
    return mandatory


def _record_occurrence(
    records: list[TokenRecord],
    token_ids: list[int],
    calibration: CalibrationRecord,
    target: bool,
) -> None:
    seen: set[int] = set()
    for token_id in token_ids:
        if not 0 <= token_id < len(records):
            raise ValidationFailure(f"tokenizer produced out-of-range ID {token_id}")
        record = records[token_id]
        if target:
            record.target_count += 1
            record.selection_reasons.add("observed_target")
        else:
            record.source_count += 1
            record.selection_reasons.add("observed_source")
        if token_id not in seen:
            record.document_count += 1
            seen.add(token_id)
        language = calibration.target_language if target else calibration.source_language
        if language:
            record.observed_languages.add(language)
        if calibration.domain:
            record.observed_domains.add(calibration.domain)
        if calibration.critical:
            record.critical_count += 1
            record.selection_reasons.add("critical_example")


def observe_calibration(
    records: list[TokenRecord],
    tokenizer: OriginalTokenizer,
    paths: Iterable[Path],
) -> None:
    """Stream source/target examples and annotate independent occurrence counts."""

    for path in paths:
        for calibration in stream_jsonl(path):
            _record_occurrence(
                records,
                tokenizer.encode(calibration.source, add_special_tokens=True),
                calibration,
                target=False,
            )
            if calibration.target is not None:
                _record_occurrence(
                    records,
                    tokenizer.encode(calibration.target, add_special_tokens=True),
                    calibration,
                    target=True,
                )


def observe_critical_terms(
    records: list[TokenRecord], tokenizer: OriginalTokenizer, paths: Iterable[Path]
) -> None:
    """Tokenize critical terms using the unchanged original tokenizer."""

    for path in paths:
        try:
            terms = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ConfigurationError(f"cannot read critical terms file {path}: {exc}") from exc
        for term in (value.strip() for value in terms):
            if not term or term.startswith("#"):
                continue
            for token_id in tokenizer.encode(term, add_special_tokens=True):
                if not 0 <= token_id < len(records):
                    raise ValidationFailure(f"tokenizer produced out-of-range ID {token_id}")
                records[token_id].critical_count += 1
                records[token_id].selection_reasons.add("critical_term")


def _add_profile_reasons(record: TokenRecord, profile: ProfileConfig) -> None:
    visible = visible_piece(record.piece)
    scripts = record.unicode_scripts - {"Common", "Inherited"}
    allowed_scripts = set(profile.unicode_scripts)
    if visible and scripts and scripts <= allowed_scripts:
        record.selection_reasons.add("allowed_unicode_script")
        if len(visible) == 1 and profile.keep_single_character_pieces_for_allowed_scripts:
            record.selection_reasons.add("allowed_script_single_character")
        if scripts == {"Latin", "Devanagari"} and profile.allow_code_switching:
            record.selection_reasons.add("code_switching_piece")
        if scripts == {"Latin"} and profile.allow_romanized_hindi:
            record.selection_reasons.add("romanized_hindi_compatible")
    if (
        record.unicode_scripts
        and record.unicode_scripts <= {"Common", "Inherited"}
        and (profile.allow_common_script or profile.allow_inherited_script)
    ):
        record.selection_reasons.add("shared_unicode_script")
    categories = record.unicode_categories
    if profile.allow_punctuation and any(category.startswith("P") for category in categories):
        record.selection_reasons.add("punctuation")
    if profile.allow_currency_symbols and "Sc" in categories:
        record.selection_reasons.add("currency_symbol")
    if profile.allow_math_symbols and "Sm" in categories:
        record.selection_reasons.add("math_symbol")
    if (profile.allow_ascii_digits or profile.allow_native_digits) and any(
        category.startswith("N") for category in categories
    ):
        record.selection_reasons.add("number")
    if profile.allow_emoji and record.has_emoji:
        record.selection_reasons.add("emoji")
    if (profile.allow_urls or profile.allow_emails) and is_url_email_building_piece(visible):
        record.selection_reasons.add("url_email_building_block")


def select_profile(
    records: list[TokenRecord],
    tokenizer: OriginalTokenizer,
    profile: ProfileConfig,
    model_config: Any | None = None,
    generation_config: Any | None = None,
) -> SelectionResult:
    """Select a conservative retained union and annotate every decision."""

    if [record.original_id for record in records] != list(range(len(records))):
        raise ValidationFailure("inventory IDs must be contiguous and ordered")
    mandatory = mandatory_token_ids(
        tokenizer, model_config, generation_config, profile.application_reserved_tokens
    )
    for token_id in mandatory:
        records[token_id].selection_reasons.add("mandatory")
    for record in records:
        if profile.keep_all_special_tokens and record.is_special:
            record.selection_reasons.add("special_token")
        if profile.keep_all_control_tokens and record.is_control:
            record.selection_reasons.add("control_token")
        if profile.keep_all_extra_ids and record.is_extra_id:
            record.selection_reasons.add("extra_id")
        if profile.keep_unknown_token and record.is_unknown:
            record.selection_reasons.add("unknown_token")
        _add_profile_reasons(record, profile)

    by_piece = {record.piece: record.original_id for record in records}
    for token_id in profile.manual_keep_ids:
        if not 0 <= token_id < len(records):
            raise ConfigurationError(f"manual_keep_ids contains out-of-range ID {token_id}")
        records[token_id].selection_reasons.add("manual_keep_id")
    for piece in profile.manual_keep_pieces:
        if piece not in by_piece:
            raise ConfigurationError(f"manual_keep_pieces contains unknown piece {piece!r}")
        records[by_piece[piece]].selection_reasons.add("manual_keep_piece")

    removals = set(profile.manual_remove_ids)
    for piece in profile.manual_remove_pieces:
        if piece not in by_piece:
            raise ConfigurationError(f"manual_remove_pieces contains unknown piece {piece!r}")
        removals.add(by_piece[piece])
    for token_id in removals:
        if not 0 <= token_id < len(records):
            raise ConfigurationError(f"manual_remove_ids contains out-of-range ID {token_id}")
        record = records[token_id]
        protected_reasons = {"mandatory", "critical_term", "critical_example"}
        if record.selection_reasons & protected_reasons:
            raise ValidationFailure(
                f"manual removal cannot remove protected ID {token_id}: "
                f"{sorted(record.selection_reasons & protected_reasons)}"
            )
        if profile.strict and (record.source_count or record.target_count):
            raise ValidationFailure(f"strict profile cannot remove observed ID {token_id}")
        record.selection_reasons.clear()
        record.risk_notes.append("manually removed; token reachability is not disproven")

    retained: list[int] = []
    excluded: list[int] = []
    for record in records:
        if record.selection_reasons:
            retained.append(record.original_id)
            if (
                "mandatory" in record.selection_reasons
                or {
                    "special_token",
                    "control_token",
                    "extra_id",
                    "unknown_token",
                }
                & record.selection_reasons
            ):
                record.tier = "mandatory_core"
            elif {"critical_term", "critical_example"} & record.selection_reasons:
                record.tier = "domain_profile"
            elif {"observed_source", "observed_target"} & record.selection_reasons:
                record.tier = "language_profile"
            else:
                record.tier = "shared_core"
        else:
            excluded.append(record.original_id)
            record.tier = "cold_fallback"
            record.risk_notes.append(
                "unselected but not certified removable; absence is not proof of unreachability"
            )
    return SelectionResult(tuple(retained), tuple(excluded), tuple(records))
