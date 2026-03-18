"""Tests for training helper functions in lib/spark-common.sh.

All tests source spark-common.sh via subprocess so they exercise the
actual shell functions exactly as consuming scripts would see them.
"""

import os
import subprocess
import tempfile

import pytest

from helpers import REPO_ROOT

SPARK_COMMON = os.path.join(REPO_ROOT, "lib", "spark-common.sh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a bash snippet that first sources spark-common.sh."""
    full_script = f'source "{SPARK_COMMON}"\n{script}'
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", full_script],
        capture_output=True,
        text=True,
        env=run_env,
    )


# ---------------------------------------------------------------------------
# 1. All 5 functions are defined after sourcing
# ---------------------------------------------------------------------------


class TestFunctionsDefined:
    EXPECTED_FUNCTIONS = [
        "spark_load_training_config",
        "spark_training_running",
        "spark_training_pid",
        "spark_apply_oom_protection",
        "spark_inference_running",
    ]

    @pytest.mark.parametrize("func", EXPECTED_FUNCTIONS)
    def test_function_defined(self, func):
        """Each training helper must be defined after sourcing spark-common.sh."""
        result = bash(f"declare -F {func} >/dev/null 2>&1")
        assert result.returncode == 0, (
            f"Function '{func}' is not defined after sourcing spark-common.sh"
        )


# ---------------------------------------------------------------------------
# 2. Global constants are set at source-time
# ---------------------------------------------------------------------------


class TestGlobalVariables:
    def test_spark_training_pid_is_constant(self):
        """SPARK_TRAINING_PID must be /tmp/spark-train.pid."""
        result = bash('echo "$SPARK_TRAINING_PID"')
        assert result.returncode == 0
        assert result.stdout.strip() == "/tmp/spark-train.pid"

    def test_spark_training_env_contains_training_env(self):
        """SPARK_TRAINING_ENV must reference a path ending in training.env."""
        result = bash('echo "$SPARK_TRAINING_ENV"')
        assert result.returncode == 0
        assert result.stdout.strip().endswith("training.env"), (
            f"Expected SPARK_TRAINING_ENV to end with 'training.env', "
            f"got: {result.stdout.strip()!r}"
        )

    def test_spark_training_log_dir_contains_training(self):
        """SPARK_TRAINING_LOG_DIR must contain 'training' in the path."""
        result = bash('echo "$SPARK_TRAINING_LOG_DIR"')
        assert result.returncode == 0
        assert "training" in result.stdout.strip(), (
            f"Expected 'training' in SPARK_TRAINING_LOG_DIR, "
            f"got: {result.stdout.strip()!r}"
        )


# ---------------------------------------------------------------------------
# 3. spark_load_training_config — missing file exits non-zero with guidance
# ---------------------------------------------------------------------------


class TestLoadTrainingConfig:
    def test_fails_when_training_env_missing(self):
        """spark_load_training_config must exit non-zero if training.env absent."""
        result = bash(
            'SPARK_TRAINING_ENV="/nonexistent/path/training.env" '
            'spark_load_training_config',
        )
        assert result.returncode != 0, (
            "Expected non-zero exit when training.env is missing"
        )

    def test_error_message_mentions_spark_train_setup(self):
        """Error message must mention 'spark-train-setup' for user guidance."""
        result = bash(
            'SPARK_TRAINING_ENV="/nonexistent/path/training.env" '
            'spark_load_training_config 2>&1 || true',
        )
        combined = result.stdout + result.stderr
        assert "spark-train-setup" in combined, (
            f"Expected 'spark-train-setup' in error output, got: {combined!r}"
        )

    def test_loads_and_exports_required_vars(self):
        """spark_load_training_config must export all 5 training variables."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as f:
            f.write(
                "TRAINING_SCRIPT_DIR=/opt/train\n"
                "TRAINING_DATA_DIR=/data/train\n"
                "TRAINING_CHECKPOINT_DIR=/ckpt\n"
                "TRAINING_MEMORY_MAX=50G\n"
                "CHECKPOINT_TIMEOUT=60\n"
            )
            env_path = f.name

        try:
            result = bash(
                f'SPARK_TRAINING_ENV="{env_path}" spark_load_training_config\n'
                'echo "SCRIPT=$TRAINING_SCRIPT_DIR"\n'
                'echo "DATA=$TRAINING_DATA_DIR"\n'
                'echo "CKPT=$TRAINING_CHECKPOINT_DIR"\n'
                'echo "MEM=$TRAINING_MEMORY_MAX"\n'
                'echo "TIMEOUT=$CHECKPOINT_TIMEOUT"\n'
            )
            assert result.returncode == 0, f"Unexpected failure: {result.stderr}"
            assert "SCRIPT=/opt/train" in result.stdout
            assert "DATA=/data/train" in result.stdout
            assert "CKPT=/ckpt" in result.stdout
            assert "MEM=50G" in result.stdout
            assert "TIMEOUT=60" in result.stdout
        finally:
            os.unlink(env_path)

    def test_applies_defaults_for_optional_vars(self):
        """TRAINING_MEMORY_MAX and CHECKPOINT_TIMEOUT default when omitted."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as f:
            f.write(
                "TRAINING_SCRIPT_DIR=/opt/train\n"
                "TRAINING_DATA_DIR=/data/train\n"
                "TRAINING_CHECKPOINT_DIR=/ckpt\n"
            )
            env_path = f.name

        try:
            result = bash(
                f'SPARK_TRAINING_ENV="{env_path}" spark_load_training_config\n'
                'echo "MEM=$TRAINING_MEMORY_MAX"\n'
                'echo "TIMEOUT=$CHECKPOINT_TIMEOUT"\n'
            )
            assert result.returncode == 0, f"Unexpected failure: {result.stderr}"
            assert "MEM=100G" in result.stdout
            assert "TIMEOUT=120" in result.stdout
        finally:
            os.unlink(env_path)

    def test_fails_when_required_var_missing(self):
        """spark_load_training_config must exit non-zero if a required var is absent."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as f:
            # Missing TRAINING_DATA_DIR and TRAINING_CHECKPOINT_DIR
            f.write("TRAINING_SCRIPT_DIR=/opt/train\n")
            env_path = f.name

        try:
            result = bash(
                f'SPARK_TRAINING_ENV="{env_path}" spark_load_training_config'
            )
            assert result.returncode != 0, (
                "Expected non-zero exit when required vars are missing"
            )
        finally:
            os.unlink(env_path)


# ---------------------------------------------------------------------------
# 4. spark_training_running — PID file behaviour
# ---------------------------------------------------------------------------


class TestTrainingRunning:
    def test_returns_1_when_no_pid_file(self):
        """spark_training_running returns 1 (not running) with no PID file."""
        result = bash(
            'SPARK_TRAINING_PID="/tmp/spark-train-test-$$.pid" '
            'spark_training_running'
        )
        assert result.returncode == 1

    def test_returns_0_when_pid_alive(self):
        """spark_training_running returns 0 when PID file holds a live PID."""
        # Use current bash PID ($$) — guaranteed alive
        result = bash(
            'PID_FILE="/tmp/spark-train-test-$$.pid"\n'
            'SPARK_TRAINING_PID="$PID_FILE"\n'
            'echo $$ > "$PID_FILE"\n'
            'spark_training_running\n'
            'rc=$?\n'
            'rm -f "$PID_FILE"\n'
            'exit $rc\n'
        )
        assert result.returncode == 0

    def test_returns_1_and_cleans_stale_pid(self):
        """spark_training_running returns 1 and removes file for dead PID."""
        # PID 99999999 is virtually guaranteed to not exist
        result = bash(
            'PID_FILE="/tmp/spark-train-stale-$$.pid"\n'
            'SPARK_TRAINING_PID="$PID_FILE"\n'
            'echo 99999999 > "$PID_FILE"\n'
            'spark_training_running\n'
            'rc=$?\n'
            # File should be gone now (cleaned up by function)
            '[[ ! -f "$PID_FILE" ]] || { echo "STALE FILE NOT REMOVED"; exit 2; }\n'
            'exit $rc\n'
        )
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# 5. spark_training_pid
# ---------------------------------------------------------------------------


class TestTrainingPid:
    def test_returns_1_when_not_running(self):
        """spark_training_pid returns 1 when training is not running."""
        result = bash(
            'SPARK_TRAINING_PID="/tmp/spark-train-test-$$.pid" '
            'spark_training_pid'
        )
        assert result.returncode == 1

    def test_outputs_pid_when_running(self):
        """spark_training_pid prints the PID to stdout when running."""
        result = bash(
            'PID_FILE="/tmp/spark-train-test-$$.pid"\n'
            'SPARK_TRAINING_PID="$PID_FILE"\n'
            'MYPID=$$\n'
            'echo $MYPID > "$PID_FILE"\n'
            'OUT=$(spark_training_pid)\n'
            'rc=$?\n'
            'rm -f "$PID_FILE"\n'
            '[[ "$OUT" == "$MYPID" ]] && exit $rc || exit 1\n'
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# 6. spark_training_write_pid / spark_training_remove_pid
# ---------------------------------------------------------------------------


class TestPidFileHelpers:
    def test_write_pid_creates_file(self):
        """spark_training_write_pid must create the PID file with given value."""
        result = bash(
            'PID_FILE="/tmp/spark-train-write-$$.pid"\n'
            'SPARK_TRAINING_PID="$PID_FILE"\n'
            'spark_training_write_pid 12345\n'
            'content=$(cat "$PID_FILE")\n'
            'rm -f "$PID_FILE"\n'
            '[[ "$content" == "12345" ]]\n'
        )
        assert result.returncode == 0

    def test_remove_pid_deletes_file(self):
        """spark_training_remove_pid must delete the PID file."""
        result = bash(
            'PID_FILE="/tmp/spark-train-remove-$$.pid"\n'
            'SPARK_TRAINING_PID="$PID_FILE"\n'
            'touch "$PID_FILE"\n'
            'spark_training_remove_pid\n'
            '[[ ! -f "$PID_FILE" ]]\n'
        )
        assert result.returncode == 0
