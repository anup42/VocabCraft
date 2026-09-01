from pathlib import Path

import pytest

from vocabcraft.config import load_config
from vocabcraft.exceptions import ConfigurationError


def test_load_en_hi_config() -> None:
    config = load_config(Path("configs/profiles/en-hi.yaml"))
    assert config.project.name == "VocabCraft"
    assert config.model.id == "google/mt5-small"
    assert config.profile.languages == ["en", "hi"]
    assert config.profile.calibration_files[0].name == "calibration-en-hi.jsonl"


def test_rejects_conflicting_manual_ids(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
project: {name: VocabCraft}
model: {id: local-model}
profile:
  id: test
  description: test profile
  languages: []
  unicode_scripts: []
  manual_keep_ids: [7]
  manual_remove_ids: [7]
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="both manually kept and removed"):
        load_config(config)


def test_rejects_renamed_project(tmp_path: Path) -> None:
    config = tmp_path / "bad-name.yaml"
    config.write_text(
        """
project: {name: WrongName}
model: {id: local-model}
profile: {id: test, description: test profile, languages: [], unicode_scripts: []}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="must be VocabCraft"):
        load_config(config)


def test_loads_local_external_evaluation_config(tmp_path: Path) -> None:
    config = tmp_path / "external.yaml"
    config.write_text(
        """
project: {name: VocabCraft}
model: {id: local-model}
profile: {id: test, description: test profile, languages: [], unicode_scripts: []}
external_evaluation:
  command: [python, private_eval.py, --model, "{model_path}"]
  metrics_file: metrics.json
  timeout_seconds: 60
  non_inferiority: {metric: task_score, maximum_allowed_drop: 0.005}
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded.external_evaluation is not None
    assert loaded.external_evaluation.metrics_file == tmp_path / "metrics.json"
    assert loaded.external_evaluation.non_inferiority.metric == "task_score"
