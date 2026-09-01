"""Fail-closed token coverage decisions and fallback dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from vocabcraft.exceptions import MissingTokenError, UnsupportedModeError
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer

FallbackPolicyName = Literal["full_model", "reject", "callback"]


@dataclass(frozen=True)
class GuardDecision:
    """Structured result of checking original IDs against an active profile."""

    supported_by_profile: bool
    original_ids: list[int]
    compact_ids: list[int] | None
    missing_original_ids: list[int]
    missing_pieces: list[str]
    fallback_required: bool
    fallback_policy: str
    fallback_reason: str | None
    profile_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable decision."""

        return asdict(self)


@dataclass(frozen=True)
class GuardedBatch:
    """Row-wise guard decisions with the original tokenizer attention mask."""

    decisions: list[GuardDecision]
    attention_mask: list[list[int]]


def guard_original_ids(
    original_ids: list[int],
    mapping: IdMapping,
    tokenizer: OriginalTokenizer,
    profile_id: str,
    fallback_policy: FallbackPolicyName,
) -> GuardDecision:
    """Map a fully covered sequence or require explicit fallback."""

    missing = sorted({token_id for token_id in original_ids if token_id not in mapping.old_to_new})
    if missing:
        pieces: list[str] = []
        for token_id in missing:
            converted = tokenizer.convert_ids_to_tokens(token_id)
            pieces.append(converted if isinstance(converted, str) else str(converted))
        return GuardDecision(
            supported_by_profile=False,
            original_ids=list(original_ids),
            compact_ids=None,
            missing_original_ids=missing,
            missing_pieces=pieces,
            fallback_required=True,
            fallback_policy=fallback_policy,
            fallback_reason="one or more original token IDs are absent from the active profile",
            profile_id=profile_id,
        )
    return GuardDecision(
        supported_by_profile=True,
        original_ids=list(original_ids),
        compact_ids=mapping.map_original_ids(original_ids),
        missing_original_ids=[],
        missing_pieces=[],
        fallback_required=False,
        fallback_policy=fallback_policy,
        fallback_reason=None,
        profile_id=profile_id,
    )


class ProfiledTokenizer:
    """Wrap the unchanged tokenizer with guard-before-map behavior."""

    def __init__(
        self,
        tokenizer: OriginalTokenizer,
        mapping: IdMapping,
        profile_id: str,
        fallback_policy: FallbackPolicyName,
    ) -> None:
        self.original_tokenizer = tokenizer
        self.mapping = mapping
        self.profile_id = profile_id
        self.fallback_policy = fallback_policy

    def encode(self, text: str, *, add_special_tokens: bool = True) -> GuardDecision:
        """Tokenize to original IDs, check coverage, then optionally map."""

        original_ids = self.original_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return guard_original_ids(
            original_ids,
            self.mapping,
            self.original_tokenizer,
            self.profile_id,
            self.fallback_policy,
        )

    def guard_batch(self, original_ids: list[list[int]]) -> list[GuardDecision]:
        """Guard an already padded or unpadded batch row by row."""

        return [
            guard_original_ids(
                row,
                self.mapping,
                self.original_tokenizer,
                self.profile_id,
                self.fallback_policy,
            )
            for row in original_ids
        ]

    def batch_encode(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> GuardedBatch:
        """Tokenize/pad a batch originally, preserve masks, then guard each complete row."""

        encoded = self.original_tokenizer(
            texts,
            padding=padding,
            add_special_tokens=add_special_tokens,
            return_attention_mask=True,
        )
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(input_ids, list) or not all(
            isinstance(row, list) and all(isinstance(value, int) for value in row)
            for row in input_ids
        ):
            raise ValueError("original tokenizer batch input_ids must be integer lists")
        if not isinstance(attention_mask, list) or not all(
            isinstance(row, list) and all(isinstance(value, int) for value in row)
            for row in attention_mask
        ):
            raise ValueError("original tokenizer attention_mask must be integer lists")
        if len(input_ids) != len(attention_mask) or any(
            len(ids) != len(mask) for ids, mask in zip(input_ids, attention_mask, strict=True)
        ):
            raise ValueError("batch input IDs and attention masks have inconsistent shapes")
        return GuardedBatch(self.guard_batch(input_ids), attention_mask)

    def decode(self, compact_ids: list[int], **kwargs: Any) -> str:
        """Map compact IDs back and decode with the unchanged tokenizer."""

        return self.original_tokenizer.decode(self.mapping.map_compact_ids(compact_ids), **kwargs)


class FallbackExecutor:
    """Dispatch only explicitly configured fallback behavior."""

    def __init__(
        self,
        policy: FallbackPolicyName,
        full_model: Callable[..., Any] | None = None,
        callback: Callable[..., Any] | None = None,
    ) -> None:
        self.policy = policy
        self.full_model = full_model
        self.callback = callback

    def execute(self, decision: GuardDecision, *args: Any, **kwargs: Any) -> Any:
        """Execute fallback or fail closed; supported decisions are rejected as misuse."""

        if not decision.fallback_required:
            raise UnsupportedModeError("fallback executor received a supported profile decision")
        if self.policy == "full_model" and self.full_model is not None:
            return self.full_model(*args, **kwargs)
        if self.policy == "callback" and self.callback is not None:
            return self.callback(decision, *args, **kwargs)
        if self.policy == "reject":
            raise MissingTokenError(decision.missing_original_ids)
        raise UnsupportedModeError(
            f"fallback policy {self.policy!r} is unavailable; request rejected"
        )
