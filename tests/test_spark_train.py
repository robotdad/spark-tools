"""Tests for spark-tools/bin/spark-train command.

Validates: file exists, is executable, passes bash syntax check,
-h shows short help, --help shows long help.
"""

import os
import subprocess

import pytest

from helpers import REPO_ROOT

BIN = os.path.join(REPO_ROOT, "bin", "spark-train")


# ---------------------------------------------------------------------------
# 1. File exists and is executable
# ---------------------------------------------------------------------------


class TestSparkTrainExists:
    def test_file_exists(self):
        """spark-train must exist in bin/."""
        assert os.path.isfile(BIN), f"spark-train must exist at {BIN}"

    def test_file_is_executable(self):
        """spark-train must be executable."""
        assert os.access(BIN, os.X_OK), f"spark-train must be executable: {BIN}"


# ---------------------------------------------------------------------------
# 2. Bash syntax check
# ---------------------------------------------------------------------------


class TestSparkTrainSyntax:
    def test_bash_syntax_ok(self):
        """bash -n must report no syntax errors."""
        result = subprocess.run(
            ["bash", "-n", BIN],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"bash -n reported syntax errors:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 3. Short help (-h)
# ---------------------------------------------------------------------------


class TestSparkTrainShortHelp:
    def test_short_help_exits_zero(self):
        """spark-train -h must exit 0."""
        result = subprocess.run([BIN, "-h"], capture_output=True, text=True)
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"

    def test_short_help_shows_usage(self):
        """spark-train -h must show Usage: line."""
        result = subprocess.run([BIN, "-h"], capture_output=True, text=True)
        assert "Usage: spark-train" in result.stdout, (
            f"Expected 'Usage: spark-train' in output:\n{result.stdout}"
        )

    def test_short_help_mentions_dry_run(self):
        """spark-train -h must mention --dry-run."""
        result = subprocess.run([BIN, "-h"], capture_output=True, text=True)
        assert "--dry-run" in result.stdout, (
            f"Expected '--dry-run' in short help:\n{result.stdout}"
        )

    def test_short_help_mentions_resume(self):
        """spark-train -h must mention --resume."""
        result = subprocess.run([BIN, "-h"], capture_output=True, text=True)
        assert "--resume" in result.stdout, (
            f"Expected '--resume' in short help:\n{result.stdout}"
        )

    def test_short_help_mentions_long_help(self):
        """spark-train -h must reference --help for the extended reference."""
        result = subprocess.run([BIN, "-h"], capture_output=True, text=True)
        assert "--help" in result.stdout, (
            f"Expected '--help' reference in short help:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# 4. Long help (--help)
# ---------------------------------------------------------------------------


class TestSparkTrainLongHelp:
    def test_long_help_exits_zero(self):
        """spark-train --help must exit 0."""
        result = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"

    def test_long_help_starts_with_spark_train(self):
        """spark-train --help must start with 'spark-train —'."""
        result = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        assert result.stdout.startswith("spark-train"), (
            f"Expected output to start with 'spark-train':\n{result.stdout[:200]}"
        )

    def test_long_help_mentions_preflight(self):
        """spark-train --help must describe the pre-flight sequence."""
        result = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        assert "Pre-flight" in result.stdout or "pre-flight" in result.stdout, (
            f"Expected 'pre-flight' in long help:\n{result.stdout[:500]}"
        )

    def test_long_help_mentions_torchrun(self):
        """spark-train --help must mention torchrun."""
        result = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        assert "torchrun" in result.stdout, (
            f"Expected 'torchrun' in long help:\n{result.stdout[:500]}"
        )

    def test_long_help_mentions_see_also(self):
        """spark-train --help must have a See also section."""
        result = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        assert "See also" in result.stdout, (
            f"Expected 'See also' in long help:\n{result.stdout}"
        )

    def test_long_help_mentions_short_help(self):
        """spark-train --help must refer back to -h for short help."""
        result = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        assert "-h" in result.stdout, (
            f"Expected '-h' reference in long help:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# 5. Script structure — sources spark-common.sh via symlink-safe boilerplate
# ---------------------------------------------------------------------------


class TestSparkTrainStructure:
    @pytest.fixture
    def source(self):
        with open(BIN) as f:
            return f.read()

    def test_uses_bash_source_symlink_boilerplate(self, source):
        """Script must use BASH_SOURCE symlink-safe sourcing."""
        assert 'SOURCE="${BASH_SOURCE[0]}"' in source, (
            "Expected symlink-safe SOURCE boilerplate"
        )

    def test_sources_spark_common(self, source):
        """Script must source spark-common.sh."""
        assert "spark-common.sh" in source, "Expected source of spark-common.sh"

    def test_calls_spark_load_config(self, source):
        """Script must call spark_load_config."""
        assert "spark_load_config" in source, "Expected call to spark_load_config"

    def test_calls_spark_load_training_config(self, source):
        """Script must call spark_load_training_config."""
        assert "spark_load_training_config" in source, (
            "Expected call to spark_load_training_config"
        )

    def test_never_hardcodes_dyad(self, source):
        """Script must never hardcode 'dyad' — use SPARK_SECONDARY_HOST."""
        # Exclude comment lines
        code_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        ]
        code = "\n".join(code_lines)
        assert "dyad" not in code, (
            "Script hardcodes 'dyad' — must use SPARK_SECONDARY_HOST instead"
        )

    def test_has_dry_run_flag(self, source):
        """Script must support --dry-run flag."""
        assert "--dry-run" in source, "Expected --dry-run flag handling"

    def test_has_resume_flag(self, source):
        """Script must support --resume flag."""
        assert "--resume" in source, "Expected --resume flag handling"

    def test_has_resume_from_step_flag(self, source):
        """Script must support --resume-from-step flag."""
        assert "--resume-from-step" in source, (
            "Expected --resume-from-step flag handling"
        )

    def test_uses_nohup_for_background(self, source):
        """Script must use nohup for background launch."""
        assert "nohup" in source, "Expected nohup for background process launch"

    def test_uses_systemd_run_memory_jail(self, source):
        """Script must use systemd-run with MemoryMax for cgroup jail."""
        assert "systemd-run" in source and "MemoryMax" in source, (
            "Expected systemd-run --scope -p MemoryMax cgroup jail"
        )

    def test_writes_pid_file(self, source):
        """Script must call spark_training_write_pid."""
        assert "spark_training_write_pid" in source, (
            "Expected call to spark_training_write_pid"
        )

    def test_checks_training_not_already_running(self, source):
        """Script must check spark_training_running before launch."""
        assert "spark_training_running" in source, (
            "Expected check for spark_training_running"
        )

    def test_checks_inference_not_running(self, source):
        """Script must check spark_inference_running before launch."""
        assert "spark_inference_running" in source, (
            "Expected check for spark_inference_running"
        )

    def test_uses_spark_ssh_for_secondary(self, source):
        """Script must use spark_ssh for secondary node communication."""
        assert "spark_ssh" in source, "Expected spark_ssh for secondary node"
