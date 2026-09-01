"""VocabCraft: safe, profile-aware vocabulary compaction."""

from vocabcraft.config import VocabCraftConfig, load_config
from vocabcraft.exceptions import VocabCraftError

__all__ = ["VocabCraftConfig", "VocabCraftError", "load_config"]
__version__ = "0.1.0"
