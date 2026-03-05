"""Model-routing auth proxy for vLLM backends.

Routes requests to the correct backend based on the model name in the request body.
Each backend periodically reports which models it serves via /v1/models.
Falls back to least-connections routing when model is unspecified or unknown.

Usage:
    uvicorn vllm_proxy:app --host 0.0.0.0 --port 9000

Configuration via environment:
    VLLM_PROXY_SECRET       - Required bearer token
    VLLM_PROXY_SECRET_FILE  - Path to file containing the secret
                              (default: ~/.config/vllm-proxy/secret.env)
    VLLM_BACKENDS           - Comma-separated backend URLs
                              (default: http://monad:8000,http://dyad:8000)
    VLLM_HEALTH_INTERVAL    - Health check interval in seconds (default: 10)
"""

import asyncio
import json
import logging
import os
import sys

import httpx
from httpx._types import QueryParamTypes  # httpx doesn't re-export this publicly
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET = os.environ.get("VLLM_PROXY_SECRET", "")
SECRET_FILE = os.environ.get(
    "VLLM_PROXY_SECRET_FILE",
    os.path.expanduser("~/.config/vllm-proxy/secret.env"),
)

# Load secret from file if not set in environment
if not SECRET and os.path.isfile(SECRET_FILE):
    with open(SECRET_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("VLLM_PROXY_SECRET="):
                SECRET = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not SECRET:
    print(
        "ERROR: No secret configured. Set VLLM_PROXY_SECRET or create secret file.",
        file=sys.stderr,
    )
    sys.exit(1)

BACKENDS_RAW = os.environ.get("VLLM_BACKENDS", "http://monad:8000,http://dyad:8000")
HEALTH_INTERVAL = int(os.environ.get("VLLM_HEALTH_INTERVAL", "10"))

logger = logging.getLogger("vllm_proxy")

# ---------------------------------------------------------------------------
# Hop-by-hop headers to strip when proxying
# ---------------------------------------------------------------------------

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def filter_headers(headers: dict) -> dict:
    """Drop hop-by-hop headers, host, and authorization."""
    filtered = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS or lk in ("host", "authorization"):
            continue
        filtered[k] = v
    return filtered


# ---------------------------------------------------------------------------
# Backend tracking
# ---------------------------------------------------------------------------


class Backend:
    """Tracks a single vLLM backend's state and models."""

    def __init__(self, url: str) -> None:
        self.url: str = url
        self.client: httpx.AsyncClient = httpx.AsyncClient(base_url=url, timeout=None)
        self.healthy: bool = False
        self.active: int = 0
        self.models: list[dict] = []  # Raw model objects from /v1/models
        self.model_ids: set[str] = set()  # Model IDs this backend serves


backends: list[Backend] = [
    Backend(u.strip()) for u in BACKENDS_RAW.split(",") if u.strip()
]

# Model-to-backend routing table (rebuilt on each health check)
model_routing: dict[str, list[Backend]] = {}


def rebuild_model_routing() -> None:
    """Rebuild the model -> [backends] routing table from current state."""
    global model_routing
    table: dict[str, list[Backend]] = {}
    for b in backends:
        if not b.healthy:
            continue
        for model_id in b.model_ids:
            table.setdefault(model_id, []).append(b)
    model_routing = table
    if table:
        models_summary = {m: [b.url for b in bs] for m, bs in table.items()}
        logger.debug("Model routing table: %s", models_summary)


# ---------------------------------------------------------------------------
# Health checking + model discovery
# ---------------------------------------------------------------------------


async def health_check_loop() -> None:
    """Periodically check health and discover models on each backend."""
    while True:
        changed = False
        for backend in backends:
            was_healthy = backend.healthy

            # Health check
            try:
                resp = await backend.client.get("/health", timeout=5.0)
                backend.healthy = resp.status_code == 200
            except Exception:
                logger.debug("Health check failed for %s", backend.url, exc_info=True)
                backend.healthy = False

            if was_healthy != backend.healthy:
                state = "healthy" if backend.healthy else "unhealthy"
                logger.info("Backend %s is now %s", backend.url, state)
                changed = True

            # Model discovery (only for healthy backends)
            if backend.healthy:
                try:
                    resp = await backend.client.get("/v1/models", timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        backend.models = data.get("data", [])
                        new_ids = {m["id"] for m in backend.models if "id" in m}
                        if new_ids != backend.model_ids:
                            backend.model_ids = new_ids
                            changed = True
                            logger.info(
                                "Backend %s serves models: %s",
                                backend.url,
                                sorted(new_ids),
                            )
                except Exception:
                    logger.debug(
                        "Model discovery failed for %s",
                        backend.url,
                        exc_info=True,
                    )
            else:
                if backend.model_ids:
                    backend.model_ids = set()
                    backend.models = []
                    changed = True

        if changed:
            rebuild_model_routing()

        await asyncio.sleep(HEALTH_INTERVAL)


# ---------------------------------------------------------------------------
# Routing — model-aware with least-connections fallback
# ---------------------------------------------------------------------------


def pick_backend(model: str | None = None) -> Backend:
    """Select backend by model name, falling back to least-connections.

    Routing priority:
    1. If model is specified and found in routing table -> pick least-busy
       backend among those serving that model.
    2. If model is not specified or not found -> pick least-busy healthy backend.
    3. If no healthy backends -> 503.
    """
    # Try model-specific routing
    if model:
        candidates = model_routing.get(model)
        if candidates:
            return min(candidates, key=lambda b: b.active)

    # Fallback: least-connections across all healthy backends
    healthy = [b for b in backends if b.healthy]
    if not healthy:
        raise HTTPException(status_code=503, detail="All backends are down")

    # If a model was requested but not found, give a helpful error
    if model and not model_routing.get(model):
        available = sorted(model_routing.keys())
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Model '{model}' not found on any backend",
                "available_models": available,
                "hint": "Use /v1/models to see available models",
            },
        )

    return min(healthy, key=lambda b: b.active)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


async def _buffered_response(
    backend: Backend,
    method: str,
    url_path: str,
    *,
    content: bytes,
    headers: dict,
    params: QueryParamTypes,
) -> JSONResponse:
    """Forward request and return a buffered JSON response."""
    resp = await backend.client.request(
        method, url_path, content=content, headers=headers, params=params
    )
    try:
        body = resp.json()
    except Exception:
        body = {"error": "non-JSON backend response"}
    return JSONResponse(status_code=resp.status_code, content=body)


async def _stream_response(
    backend: Backend,
    method: str,
    url_path: str,
    *,
    content: bytes,
    headers: dict,
    params: QueryParamTypes,
) -> StreamingResponse:
    """Forward request and return a streaming SSE response."""
    req = backend.client.build_request(
        method, url_path, content=content, headers=headers, params=params
    )
    resp = await backend.client.send(req, stream=True)

    async def _generate():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            backend.active -= 1

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="vLLM Auth Proxy", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(health_check_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    for backend in backends:
        await backend.client.aclose()


@app.get("/health")
async def health() -> JSONResponse:
    """Report backend health with model info; 200 if any healthy, 503 if all down."""
    status = [
        {
            "url": b.url,
            "healthy": b.healthy,
            "active": b.active,
            "models": sorted(b.model_ids),
        }
        for b in backends
    ]
    any_healthy = any(b.healthy for b in backends)
    return JSONResponse(
        status_code=200 if any_healthy else 503,
        content={"backends": status},
    )


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
)
async def proxy_v1(path: str, request: Request):
    # Auth check
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.body()
    headers = filter_headers(dict(request.headers))
    url_path = f"/v1/{path}"

    # Handle /v1/models specially — aggregate from all healthy backends
    if path == "models" and request.method == "GET":
        all_models = []
        seen_ids: set[str] = set()
        for b in backends:
            if not b.healthy:
                continue
            for m in b.models:
                mid = m.get("id", "")
                if mid not in seen_ids:
                    all_models.append(m)
                    seen_ids.add(mid)
        return JSONResponse(content={"object": "list", "data": all_models})

    # Parse body for model name and streaming flag
    model: str | None = None
    is_stream = False
    if body:
        try:
            parsed = json.loads(body)
            model = parsed.get("model")
            is_stream = parsed.get("stream") is True
        except (json.JSONDecodeError, AttributeError):
            pass

    backend = pick_backend(model)
    backend.active += 1
    try:
        if is_stream:
            return await _stream_response(
                backend,
                request.method,
                url_path,
                content=body,
                headers=headers,
                params=request.query_params,
            )
        else:
            return await _buffered_response(
                backend,
                request.method,
                url_path,
                content=body,
                headers=headers,
                params=request.query_params,
            )
    finally:
        if not is_stream:
            backend.active -= 1
