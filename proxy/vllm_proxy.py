"""
vLLM Auth Proxy - Bearer token authentication for vLLM API.

Lightweight reverse proxy that adds bearer-token auth in front of a vLLM
endpoint. Supports both regular and streaming (SSE) responses.

Usage:
    uvicorn vllm_proxy:app --host 0.0.0.0 --port 9000

Configuration via environment:
    VLLM_PROXY_SECRET  - Required bearer token (or reads from SECRET_FILE)
    VLLM_PROXY_SECRET_FILE - Path to file containing the secret
    VLLM_UPSTREAM_URL  - Upstream vLLM URL (default: http://localhost:8000)
"""

import os
import sys

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

# --- Configuration ---
SECRET = os.environ.get("VLLM_PROXY_SECRET", "")
SECRET_FILE = os.environ.get("VLLM_PROXY_SECRET_FILE",
    os.path.expanduser("~/.config/vllm-proxy/secret.env"))
UPSTREAM = os.environ.get("VLLM_UPSTREAM_URL", "http://localhost:8000")

# Load secret from file if not in env
if not SECRET and os.path.isfile(SECRET_FILE):
    with open(SECRET_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("VLLM_PROXY_SECRET="):
                SECRET = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not SECRET:
    print("ERROR: No secret configured. Set VLLM_PROXY_SECRET or create secret file.", file=sys.stderr)
    sys.exit(1)

app = FastAPI(title="vLLM Auth Proxy", docs_url=None, redoc_url=None)

# Shared async HTTP client
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=None)

# Headers to strip when proxying
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "authorization",
})


def verify_auth(request: Request):
    """Check bearer token."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def filter_headers(headers):
    """Remove hop-by-hop and auth headers."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


@app.get("/health")
async def health():
    """Proxy health check (no auth required)."""
    try:
        resp = await client.get("/health")
        return JSONResponse(content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"status": "ok"}, status_code=resp.status_code)
    except Exception:
        return JSONResponse(content={"status": "upstream_unreachable"}, status_code=502)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Reverse proxy with auth + streaming support."""
    verify_auth(request)

    headers = filter_headers(dict(request.headers))
    body = await request.body()

    # Check if this is a streaming request
    is_streaming = False
    if request.method == "POST" and body:
        try:
            import json
            data = json.loads(body)
            is_streaming = data.get("stream", False)
        except (json.JSONDecodeError, AttributeError):
            pass

    url = f"/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    if is_streaming:
        # Stream the response (SSE for chat completions)
        req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        upstream = await client.send(req, stream=True)

        async def stream_body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=filter_headers(dict(upstream.headers)),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )
    else:
        # Regular request/response
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        return JSONResponse(
            content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
            status_code=resp.status_code,
            headers=filter_headers(dict(resp.headers)),
        )


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
