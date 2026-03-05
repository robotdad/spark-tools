"""Shared non-fixture utilities for the spark-tools test suite.

Import with:   from helpers import REPO_ROOT, parse_env_file

(conftest.py is reserved for pytest fixtures; plain helpers live here so
 they can be imported as a normal module without pytest's conftest magic.)
"""

import os

# ---------------------------------------------------------------------------
# Repo-root resolution — works regardless of cwd when pytest is invoked
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Env-file parser
# ---------------------------------------------------------------------------


def parse_env_file(path: str) -> dict[str, str]:
    """Parse a shell-style .env file into {key: value}, skipping comments.

    Handles:
      - Blank lines and ``#``-prefixed comment lines
      - ``KEY=value``, ``KEY="value"``, ``KEY='value'``
    The last definition of a key wins (matches shell semantics).
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
