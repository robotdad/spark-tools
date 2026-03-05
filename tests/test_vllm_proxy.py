"""Tests for the spark-tools vllm_proxy with model-aware routing and streaming.

Adapted from Brian's dgx-spark-cluster test suite.  Key differences vs. the
original:

  Secret env var   : VLLM_PROXY_SECRET          (Brian used VLLM_FAMILY_SECRET)
  Secret file      : VLLM_PROXY_SECRET_FILE      (new fallback mechanism)
  No-secret exit   : sys.exit(1) → SystemExit   (Brian's proxy raised RuntimeError)
  Default backends : http://monad:8000,http://dyad:8000
  Model routing    : model_routing, rebuild_model_routing, Backend.model_ids
  /v1/models       : aggregated from all healthy backends (200 even when all down)
  Auth variable    : VLLM_PROXY_SECRET
"""

import ast
import importlib
import os
import sys

import pytest

from helpers import REPO_ROOT

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

PROXY_PATH = os.path.join(REPO_ROOT, "proxy", "vllm_proxy.py")
PROXY_DIR = os.path.dirname(PROXY_PATH)


# ---------------------------------------------------------------------------
# Helper: ensure proxy dir is on sys.path without leaking between tests
# ---------------------------------------------------------------------------


def _insert_proxy_dir():
    if PROXY_DIR not in sys.path:
        sys.path.insert(0, PROXY_DIR)


# ---------------------------------------------------------------------------
# 1. Syntax validation — the file must parse as valid Python
# ---------------------------------------------------------------------------


def test_proxy_syntax_valid():
    """ast.parse must succeed on the proxy source."""
    with open(PROXY_PATH) as f:
        source = f.read()
    ast.parse(source)  # raises SyntaxError on failure


# ---------------------------------------------------------------------------
# 2. Importability — with VLLM_PROXY_SECRET set, module must load
# ---------------------------------------------------------------------------


def test_proxy_imports_with_secret(monkeypatch, tmp_path):
    """Module loads without error when VLLM_PROXY_SECRET is set."""
    monkeypatch.setenv("VLLM_PROXY_SECRET", "test")
    # Point secret file to a nonexistent path so the env var path is exercised
    monkeypatch.setenv("VLLM_PROXY_SECRET_FILE", str(tmp_path / "no-file.env"))
    sys.modules.pop("vllm_proxy", None)
    _insert_proxy_dir()
    try:
        mod = importlib.import_module("vllm_proxy")
        assert hasattr(mod, "app"), "Module must expose a FastAPI 'app'"
    finally:
        sys.modules.pop("vllm_proxy", None)


# ---------------------------------------------------------------------------
# 3. SystemExit when VLLM_PROXY_SECRET is missing and no secret file
#    (Brian's proxy raised RuntimeError; ours calls sys.exit(1))
# ---------------------------------------------------------------------------


def test_proxy_exits_without_secret(monkeypatch, tmp_path):
    """Module must sys.exit(1) if VLLM_PROXY_SECRET is unset and no secret file."""
    monkeypatch.delenv("VLLM_PROXY_SECRET", raising=False)
    monkeypatch.setenv("VLLM_PROXY_SECRET_FILE", str(tmp_path / "no-file.env"))
    sys.modules.pop("vllm_proxy", None)
    _insert_proxy_dir()
    try:
        with pytest.raises(SystemExit) as exc_info:
            importlib.import_module("vllm_proxy")
        assert exc_info.value.code == 1, "Must exit with code 1"
    finally:
        sys.modules.pop("vllm_proxy", None)


# ---------------------------------------------------------------------------
# 4. Secret-file fallback: VLLM_PROXY_SECRET_FILE
# ---------------------------------------------------------------------------


def test_proxy_loads_secret_from_file(monkeypatch, tmp_path):
    """Module loads and exposes correct SECRET when provided via secret file."""
    monkeypatch.delenv("VLLM_PROXY_SECRET", raising=False)
    secret_file = tmp_path / "secret.env"
    secret_file.write_text('VLLM_PROXY_SECRET="file-loaded-secret"\n')
    monkeypatch.setenv("VLLM_PROXY_SECRET_FILE", str(secret_file))
    sys.modules.pop("vllm_proxy", None)
    _insert_proxy_dir()
    try:
        mod = importlib.import_module("vllm_proxy")
        assert hasattr(mod, "app")
        assert mod.SECRET == "file-loaded-secret"
    finally:
        sys.modules.pop("vllm_proxy", None)


def test_proxy_env_var_wins_over_file(monkeypatch, tmp_path):
    """VLLM_PROXY_SECRET env var takes precedence over the secret file."""
    monkeypatch.setenv("VLLM_PROXY_SECRET", "env-secret")
    secret_file = tmp_path / "secret.env"
    secret_file.write_text('VLLM_PROXY_SECRET="file-secret"\n')
    monkeypatch.setenv("VLLM_PROXY_SECRET_FILE", str(secret_file))
    sys.modules.pop("vllm_proxy", None)
    _insert_proxy_dir()
    try:
        mod = importlib.import_module("vllm_proxy")
        assert mod.SECRET == "env-secret"
    finally:
        sys.modules.pop("vllm_proxy", None)


# ---------------------------------------------------------------------------
# 5. Source-level structural checks (AST inspection)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def proxy_source():
    with open(PROXY_PATH) as f:
        return f.read()


@pytest.fixture(scope="module")
def proxy_ast(proxy_source):
    return ast.parse(proxy_source)


def _collect_names(tree) -> set[str]:
    """Collect all top-level assignment targets, function names, and class names."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


class TestASTStructure:
    def test_has_backend_class(self, proxy_ast):
        class_names = [
            n.name for n in ast.walk(proxy_ast) if isinstance(n, ast.ClassDef)
        ]
        assert "Backend" in class_names, "Must have a Backend class"

    def test_has_pick_backend_function(self, proxy_ast):
        assert "pick_backend" in _collect_names(proxy_ast)

    def test_has_health_check_loop(self, proxy_ast):
        assert "health_check_loop" in _collect_names(proxy_ast)

    def test_has_buffered_response(self, proxy_ast):
        assert "_buffered_response" in _collect_names(proxy_ast)

    def test_has_stream_response(self, proxy_ast):
        assert "_stream_response" in _collect_names(proxy_ast)

    def test_has_rebuild_model_routing(self, proxy_ast):
        """User's proxy adds model-aware routing — must have rebuild_model_routing."""
        assert "rebuild_model_routing" in _collect_names(proxy_ast)

    def test_has_health_endpoint(self, proxy_source):
        assert "/health" in proxy_source

    def test_has_catch_all_route(self, proxy_source):
        assert "/v1/{path:path}" in proxy_source

    def test_uses_fastapi_and_httpx(self, proxy_source):
        assert "FastAPI" in proxy_source
        assert "httpx" in proxy_source

    # --- Correct env var names -----------------------------------------------

    def test_env_var_vllm_proxy_secret(self, proxy_source):
        """Must read VLLM_PROXY_SECRET (not VLLM_FAMILY_SECRET)."""
        assert "VLLM_PROXY_SECRET" in proxy_source
        assert "VLLM_FAMILY_SECRET" not in proxy_source, (
            "Must use VLLM_PROXY_SECRET, not the dgx-cluster name VLLM_FAMILY_SECRET"
        )

    def test_env_var_secret_file(self, proxy_source):
        """Must support VLLM_PROXY_SECRET_FILE for file-based secret loading."""
        assert "VLLM_PROXY_SECRET_FILE" in proxy_source

    def test_env_var_backends(self, proxy_source):
        assert "VLLM_BACKENDS" in proxy_source

    def test_env_var_health_interval(self, proxy_source):
        assert "VLLM_HEALTH_INTERVAL" in proxy_source

    # --- Correct defaults ----------------------------------------------------

    def test_default_backends_monad_dyad(self, proxy_source):
        """Default backends must reference monad and dyad (not spark-1/spark-2)."""
        assert "http://monad:8000" in proxy_source
        assert "http://dyad:8000" in proxy_source

    def test_default_health_interval_10(self, proxy_source):
        assert '"10"' in proxy_source or "'10'" in proxy_source

    # --- Streaming / protocol ------------------------------------------------

    def test_streaming_detection(self, proxy_source):
        assert "stream" in proxy_source.lower()

    def test_streaming_response_headers(self, proxy_source):
        assert "no-cache" in proxy_source
        assert "X-Accel-Buffering" in proxy_source

    def test_stream_response_media_type(self, proxy_source):
        assert "text/event-stream" in proxy_source

    def test_hop_by_hop_headers_filtered(self, proxy_source):
        for header in ("connection", "keep-alive", "transfer-encoding", "upgrade"):
            assert header in proxy_source.lower(), (
                f"Must filter hop-by-hop header: {header}"
            )

    def test_bearer_auth_check(self, proxy_source):
        assert "Bearer" in proxy_source
        assert "401" in proxy_source

    def test_503_when_all_down(self, proxy_source):
        assert "503" in proxy_source

    def test_least_connections_strategy(self, proxy_source):
        """Backend selection tracks in-flight count via .active."""
        assert "active" in proxy_source

    def test_aiter_bytes_for_streaming(self, proxy_source):
        assert "aiter_bytes" in proxy_source

    def test_startup_event_launches_health_loop(self, proxy_source):
        assert "startup" in proxy_source or "lifespan" in proxy_source

    # --- Model-aware routing (user's feature) --------------------------------

    def test_model_routing_table(self, proxy_source):
        assert "model_routing" in proxy_source

    def test_model_ids_tracked_on_backend(self, proxy_source):
        """Backend must track which model IDs it serves (model_ids attribute)."""
        assert "model_ids" in proxy_source

    def test_v1_models_discovery(self, proxy_source):
        """/v1/models endpoint is used for model discovery in health loop."""
        assert "/v1/models" in proxy_source


# ---------------------------------------------------------------------------
# 6. Functional tests with HTTPX TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_proxy(monkeypatch, tmp_path):
    """Import proxy module with isolated test env vars; clean up on teardown."""
    monkeypatch.setenv("VLLM_PROXY_SECRET", "test-secret")
    monkeypatch.setenv("VLLM_PROXY_SECRET_FILE", str(tmp_path / "no-file.env"))
    monkeypatch.setenv("VLLM_BACKENDS", "http://127.0.0.1:19999")
    monkeypatch.setenv("VLLM_HEALTH_INTERVAL", "999")
    sys.modules.pop("vllm_proxy", None)
    _insert_proxy_dir()
    try:
        mod = importlib.import_module("vllm_proxy")
        yield mod
    finally:
        sys.modules.pop("vllm_proxy", None)


class TestHTTPEndpoints:
    def test_health_endpoint_returns_valid_status(self, loaded_proxy):
        """GET /health must respond 200 or 503 (all backends down in tests)."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code in (200, 503), f"Unexpected status: {resp.status_code}"

    def test_health_response_has_backends_key(self, loaded_proxy):
        """GET /health body must include a 'backends' list."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.get("/health")
        data = resp.json()
        assert "backends" in data, "Health response must contain 'backends' key"
        assert isinstance(data["backends"], list)

    def test_auth_rejected_without_token(self, loaded_proxy):
        """Requests without any Authorization header must get 401."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.post("/v1/chat/completions", json={"model": "test"})
        assert resp.status_code == 401

    def test_auth_rejected_with_wrong_token(self, loaded_proxy):
        """Requests with incorrect Bearer token must get 401."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_auth_with_correct_token_passes_to_backend(self, loaded_proxy):
        """Correct token must pass auth; all-down backends return 503 (or 404 model)."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test"},
            headers={"Authorization": "Bearer test-secret"},
        )
        # Auth passed — only backend errors remain (no 401)
        assert resp.status_code != 401, "Should not be rejected by auth"
        assert resp.status_code in (503, 404)

    def test_get_models_aggregated_returns_200(self, loaded_proxy):
        """GET /v1/models returns 200 with empty data list when all backends down."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_models_rejected_without_auth(self, loaded_proxy):
        """GET /v1/models without token must return 401."""
        from starlette.testclient import TestClient

        client = TestClient(loaded_proxy.app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        assert resp.status_code == 401


class TestBackendAttributes:
    def test_backend_has_standard_attributes(self, loaded_proxy):
        """Backend instance must have url, client, healthy, active."""
        backend = loaded_proxy.backends[0]
        assert hasattr(backend, "url")
        assert hasattr(backend, "client")
        assert hasattr(backend, "healthy")
        assert hasattr(backend, "active")
        assert isinstance(backend.healthy, bool)
        assert isinstance(backend.active, int)

    def test_backend_has_model_tracking_attributes(self, loaded_proxy):
        """User's proxy adds model_ids and models for model-aware routing."""
        backend = loaded_proxy.backends[0]
        assert hasattr(backend, "model_ids"), "Backend must have model_ids (set)"
        assert hasattr(backend, "models"), "Backend must have models (list)"
        assert isinstance(backend.model_ids, set)
        assert isinstance(backend.models, list)

    def test_backend_url_matches_env(self, loaded_proxy):
        """Backend URL must reflect VLLM_BACKENDS env var."""
        urls = [b.url for b in loaded_proxy.backends]
        assert "http://127.0.0.1:19999" in urls


# ---------------------------------------------------------------------------
# 7. Streaming resource safety (AST quality checks)
# ---------------------------------------------------------------------------


class TestStreamingSafety:
    def test_generate_uses_try_finally(self, proxy_ast):
        """_generate must use try/finally so resp.aclose() fires on disconnect."""
        for node in ast.walk(proxy_ast):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_stream_response"
            ):
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.AsyncFunctionDef)
                        and inner.name == "_generate"
                    ):
                        has_try_finally = any(
                            isinstance(n, ast.Try) and n.finalbody
                            for n in ast.walk(inner)
                        )
                        assert has_try_finally, (
                            "_generate must use try/finally to ensure resp.aclose() "
                            "is called even when the client disconnects mid-stream"
                        )
                        return
        pytest.fail("Could not find _generate nested inside _stream_response")

    def test_proxy_v1_conditionally_decrements_active(self, proxy_source):
        """proxy_v1 finally block must only decrement backend.active for non-streaming.

        Streaming responses decrement inside the generator (so the count remains
        elevated until the last byte is sent).  An unconditional decrement in the
        proxy_v1 finally block would under-count in-flight streaming requests.
        """
        tree = ast.parse(proxy_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "proxy_v1":
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Try) and inner.finalbody:
                        has_conditional = any(
                            isinstance(n, ast.If) for n in inner.finalbody
                        )
                        assert has_conditional, (
                            "proxy_v1 finally block must conditionally decrement "
                            "backend.active (not for streaming — generator handles it)"
                        )
                        return
        pytest.fail("Could not find try/finally block inside proxy_v1")

    def test_params_type_not_bare_object(self, proxy_source):
        """Type annotation 'params: object' is too loose — must use a proper type."""
        assert "params: object" not in proxy_source, (
            "params must use a proper type annotation, not bare 'object'"
        )
