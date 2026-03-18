"""Tests for spark-tools training configuration file.

Validates training.env.example contains exactly the required variables
for distributed training setup, following the same conventions as other
config files in this project.
"""

import os

import pytest

from helpers import REPO_ROOT, parse_env_file

# ---------------------------------------------------------------------------
# Config path
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.join(REPO_ROOT, "config")
TRAINING_ENV = os.path.join(CONFIG_DIR, "training.env.example")


# ---------------------------------------------------------------------------
# 1. File exists
# ---------------------------------------------------------------------------


class TestTrainingEnvFileExists:
    def test_file_exists(self):
        assert os.path.isfile(TRAINING_ENV), (
            f"training.env.example must exist at {TRAINING_ENV}"
        )


# ---------------------------------------------------------------------------
# 2. Training env variables
# ---------------------------------------------------------------------------


class TestTrainingEnvVariables:
    @pytest.fixture
    def config(self):
        assert os.path.isfile(TRAINING_ENV), f"Must exist: {TRAINING_ENV}"
        return parse_env_file(TRAINING_ENV)

    @pytest.fixture
    def raw(self):
        assert os.path.isfile(TRAINING_ENV), f"Must exist: {TRAINING_ENV}"
        with open(TRAINING_ENV) as f:
            return f.read()

    def test_has_exactly_five_variables(self, config):
        """Spec requires exactly 5 configuration variables."""
        assert len(config) == 5, (
            f"training.env.example must have exactly 5 variables, got {len(config)}: "
            f"{list(config.keys())}"
        )

    def test_has_training_script_dir(self, config):
        assert "TRAINING_SCRIPT_DIR" in config, "TRAINING_SCRIPT_DIR must be defined"

    def test_training_script_dir_is_non_empty(self, config):
        assert config["TRAINING_SCRIPT_DIR"], "TRAINING_SCRIPT_DIR must be non-empty"

    def test_has_training_data_dir(self, config):
        assert "TRAINING_DATA_DIR" in config, "TRAINING_DATA_DIR must be defined"

    def test_training_data_dir_is_non_empty(self, config):
        assert config["TRAINING_DATA_DIR"], "TRAINING_DATA_DIR must be non-empty"

    def test_has_training_checkpoint_dir(self, config):
        assert "TRAINING_CHECKPOINT_DIR" in config, (
            "TRAINING_CHECKPOINT_DIR must be defined"
        )

    def test_training_checkpoint_dir_is_non_empty(self, config):
        assert config["TRAINING_CHECKPOINT_DIR"], (
            "TRAINING_CHECKPOINT_DIR must be non-empty"
        )

    def test_has_training_memory_max(self, config):
        assert "TRAINING_MEMORY_MAX" in config, "TRAINING_MEMORY_MAX must be defined"

    def test_training_memory_max_defaults_to_100g(self, config):
        assert config["TRAINING_MEMORY_MAX"] == "100G", (
            f"TRAINING_MEMORY_MAX must default to '100G', got '{config.get('TRAINING_MEMORY_MAX')}'"
        )

    def test_has_checkpoint_timeout(self, config):
        assert "CHECKPOINT_TIMEOUT" in config, "CHECKPOINT_TIMEOUT must be defined"

    def test_checkpoint_timeout_defaults_to_120(self, config):
        assert config["CHECKPOINT_TIMEOUT"] == "120", (
            f"CHECKPOINT_TIMEOUT must default to '120', got '{config.get('CHECKPOINT_TIMEOUT')}'"
        )

    def test_all_values_non_empty(self, config):
        """Every variable must have a non-empty value."""
        for key, value in config.items():
            assert value, f"Variable {key} must have a non-empty value"

    def test_no_variable_names_reference_lampblack(self, config):
        """Variable names must be generic (no lampblack coupling)."""
        for key in config:
            assert "lampblack" not in key.lower(), (
                f"Variable name '{key}' must not reference 'lampblack'"
            )

    def test_contains_section_separators(self, raw):
        """Config file must use ===== section separators like other .env.example files."""
        assert "=====" in raw, (
            "training.env.example must contain ===== section separator comments"
        )

    def test_parses_correctly_via_helpers(self):
        """File must parse without errors via helpers.parse_env_file()."""
        result = parse_env_file(TRAINING_ENV)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_expected_variable_names_present(self, config):
        """All five expected variable names must be present."""
        expected = {
            "TRAINING_SCRIPT_DIR",
            "TRAINING_DATA_DIR",
            "TRAINING_CHECKPOINT_DIR",
            "TRAINING_MEMORY_MAX",
            "CHECKPOINT_TIMEOUT",
        }
        assert set(config.keys()) == expected, (
            f"Expected variables {expected}, got {set(config.keys())}"
        )
