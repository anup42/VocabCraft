"""Structured exceptions raised by VocabCraft."""


class VocabCraftError(Exception):
    """Base class for user-facing VocabCraft failures."""


class ConfigurationError(VocabCraftError):
    """Raised when a profile configuration is invalid."""


class InspectionError(VocabCraftError):
    """Raised when a model structure cannot be safely inspected."""


class UnsupportedModelError(InspectionError):
    """Raised when no safe adapter exists for a model structure."""


class MappingError(VocabCraftError):
    """Raised when token IDs cannot be mapped without substitution."""


class MissingTokenError(MappingError):
    """Raised when one or more original IDs are absent from a profile."""

    def __init__(self, missing_ids: list[int]) -> None:
        self.missing_ids = sorted(set(missing_ids))
        super().__init__(f"profile is missing original token IDs: {self.missing_ids}")


class ArtifactError(VocabCraftError):
    """Raised when an artifact is incomplete, incompatible, or corrupt."""


class PackConflictError(ArtifactError):
    """Raised when packs contain incompatible metadata or conflicting rows."""


class ValidationFailure(VocabCraftError):
    """Raised when a required validation invariant fails."""


class UnsupportedModeError(VocabCraftError):
    """Raised when a requested research mode is unsafe for the loaded model."""
