"""Shared fixtures and helpers for the spark-tools test suite."""

import os


# ---------------------------------------------------------------------------
# Repo-root resolution — works regardless of cwd when pytest is invoked
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Env-file parser (used by test_config_files.py)
# ---------------------------------------------------------------------------


def parse_env_file(path: str) -> dict[str, str]:
    """Parse a shell-style .env file into {key: value}, skipping comments.

    Handles:
      - Blank lines and lines starting with ``#``
      - ``KEY=value``, ``KEY="value"``, ``KEY='value'``
    """
    result: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                result[key.strip()] = value
    return result
