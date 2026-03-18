"""Tests for spark-tools training configuration file.

Validates that training.env.example contains all required variables with
correct defaults. Follows the same pattern as test_config_files.py.
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


class TestTrainingEnvExists:
    def test_file_exists(self):
        assert os.path.isfile(TRAINING_ENV), (
            f"training.env.example must exist at {TRAINING_ENV}"
        )


# ---------------------------------------------------------------------------
# 2. Required variables are present
# ---------------------------------------------------------------------------


class TestTrainingEnvVariables:
    @pytest.fixture
    def config(self):
        assert os.path.isfile(TRAINING_ENV), f"Must exist: {TRAINING_ENV}"
        return parse_env_file(TRAINING_ENV)

    @pytest.fixture
    def raw(self):
        with open(TRAINING_ENV) as f:
            return f.read()

    # All five variables must exist

    def test_has_training_script_dir(self, config):
        """Path to training project must be specified."""
        assert "TRAINING_SCRIPT_DIR" in config
        assert config["TRAINING_SCRIPT_DIR"], "TRAINING_SCRIPT_DIR must be non-empty"

    def test_has_training_data_dir(self, config):
        """Path to tokenized data shards must be specified."""
        assert "TRAINING_DATA_DIR" in config
        assert config["TRAINING_DATA_DIR"], "TRAINING_DATA_DIR must be non-empty"

    def test_has_training_checkpoint_dir(self, config):
        """Path to checkpoint output directory must be specified."""
        assert "TRAINING_CHECKPOINT_DIR" in config
        assert config["TRAINING_CHECKPOINT_DIR"], (
            "TRAINING_CHECKPOINT_DIR must be non-empty"
        )

    def test_has_training_memory_max(self, config):
        """Cgroup memory ceiling must be specified."""
        assert "TRAINING_MEMORY_MAX" in config
        assert config["TRAINING_MEMORY_MAX"], "TRAINING_MEMORY_MAX must be non-empty"

    def test_has_checkpoint_timeout(self, config):
        """Graceful stop timeout must be specified."""
        assert "CHECKPOINT_TIMEOUT" in config
        assert config["CHECKPOINT_TIMEOUT"], "CHECKPOINT_TIMEOUT must be non-empty"

    # Default values

    def test_memory_max_default_is_100g(self, config):
        """Default 100G leaves ~28GB headroom per node for OS/SSH/NCCL."""
        assert config["TRAINING_MEMORY_MAX"] == "100G"

    def test_checkpoint_timeout_default_is_120(self, config):
        """Default 120 seconds for checkpoint save after SIGTERM."""
        assert config["CHECKPOINT_TIMEOUT"] == "120"

    def test_checkpoint_timeout_is_numeric(self, config):
        """CHECKPOINT_TIMEOUT must be a pure integer (seconds)."""
        assert config["CHECKPOINT_TIMEOUT"].isdigit(), (
            "CHECKPOINT_TIMEOUT must be an integer"
        )

    # Naming conventions — generic, not project-specific

    def test_no_lampblack_coupling(self, raw):
        """Config must use generic names, not project-specific ones."""
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key = line.split("=")[0].strip()
                assert "LAMPBLACK" not in key.upper(), (
                    f"Variable name {key} must not reference lampblack"
                )

    def test_has_section_comments(self, raw):
        """Config should have section separator comments like other .env.example files."""
        assert "=====" in raw, (
            "training.env.example should use ===== section separators"
        )

    def test_exactly_five_variables(self, config):
        """Training config should have exactly 5 variables — no more, no less."""
        assert len(config) == 5, (
            f"Expected exactly 5 variables, got {len(config)}: {list(config.keys())}"
        )
