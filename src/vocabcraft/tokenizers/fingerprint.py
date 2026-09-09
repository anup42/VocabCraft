"""Tokenizer behavior fingerprints independent of local checkpoint paths."""

from __future__ import annotations

import json
from typing import Any

from vocabcraft.hashing import sha256_json


def tokenizer_behavior_hash(tokenizer: Any) -> str:
    """Hash segmentation, added tokens, special IDs, and preprocessing settings."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    serializer = getattr(backend, "to_str", None)
    if callable(serializer):
        segmentation: Any = json.loads(serializer())
        # A call with padding/truncation can mutate these backend execution settings.
        # The public wrapper sets them for each request; hash their public defaults below.
        segmentation.pop("padding", None)
        segmentation.pop("truncation", None)
    else:
        from vocabcraft.models.mt5 import _tokenizer_hash

        segmentation = {"sentencepiece_or_vocabulary_sha256": _tokenizer_hash(tokenizer)}
    settings = {
        name: getattr(tokenizer, name, None)
        for name in (
            "model_max_length",
            "padding_side",
            "truncation_side",
            "clean_up_tokenization_spaces",
            "add_prefix_space",
            "do_lower_case",
            "legacy",
            "split_special_tokens",
            "pad_token_type_id",
        )
    }
    added = {}
    for token_id, token in getattr(tokenizer, "added_tokens_decoder", {}).items():
        getstate = getattr(token, "__getstate__", None)
        added[str(token_id)] = getstate() if callable(getstate) else str(token)
    return sha256_json(
        {
            "segmentation": segmentation,
            "settings": settings,
            "added_tokens": added,
            "special_ids": list(tokenizer.all_special_ids),
            "special_tokens": list(tokenizer.all_special_tokens),
        }
    )
