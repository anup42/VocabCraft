# VocabCraft

Safe, profile-aware vocabulary compaction for multilingual models.

VocabCraft builds smaller language-specific model profiles while preserving the
original tokenizer, validating model behavior, and explicitly handling missing
tokens. The first adapter targets `google/mt5-small`; the first profile targets
English and Hindi.

The conservative runtime path never substitutes a missing token. It tokenizes
with the unchanged source tokenizer, verifies every original token ID, maps only
fully covered sequences, and otherwise invokes an explicit fallback policy.

The implementation is under active development. See the documentation and CLI
help for supported operations and the precise limits of each execution mode.

```powershell
python -m pip install -e ".[dev]"
vocabcraft --help
pytest tests/unit
```

