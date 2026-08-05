import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from legacy_tts import router as legacy_tts_router
from speaker_registration import router as speaker_registration_router
from streaming_tts import router as streaming_tts_router


SERVICE_NAME = "CosyVoice3Pro Web Gateway"
SERVICE_VERSION = "1.7.0"
TRITON_UPSTREAM = os.environ.get(
    "COSYVOICE_TRITON_UPSTREAM", "http://127.0.0.1:18100").rstrip("/")
WEB_DIR = Path(__file__).resolve().parent / "web"

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@asynccontextmanager
async def lifespan(app):
    app.state.triton_upstream = TRITON_UPSTREAM
    app.state.http_client = httpx.AsyncClient(
        timeout=None,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=256,
            max_keepalive_connections=64,
        ),
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.include_router(legacy_tts_router)
app.include_router(speaker_registration_router)
app.include_router(streaming_tts_router)


def _forward_headers(headers):
    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() not in {"host", "content-length"}
    }
    # httpx otherwise adds gzip automatically, which changes the byte-level
    # behavior seen by existing curl clients that do not request compression.
    forwarded.setdefault("accept-encoding", "identity")
    return forwarded


async def _upstream_ready(request, path):
    try:
        response = await request.app.state.http_client.get(
            f"{TRITON_UPSTREAM}{path}",
            timeout=3,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@app.get("/health")
async def public_health(request: Request):
    triton_ready, tts_ready, streaming_ready, registry_ready = await asyncio.gather(
        _upstream_ready(request, "/v2/health/ready"),
        _upstream_ready(request, "/v2/models/CosyVoice3Pro/ready"),
        _upstream_ready(
            request, "/v2/models/CosyVoice3ProStreaming/ready"),
        _upstream_ready(
            request,
            "/v2/models/CosyVoice3ProSpeakerRegistry/ready",
        ),
    )
    ready = triton_ready and tts_ready and streaming_ready and registry_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "unavailable",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "gatewayReady": True,
            "tritonReady": triton_ready,
            "models": {
                "ttsReady": tts_ready,
                "streamingTtsReady": streaming_ready,
                "speakerRegistryReady": registry_ready,
            },
        },
    )


@app.get("/admin/api/info")
async def service_info(request: Request):
    ready = False
    try:
        response = await request.app.state.http_client.get(
            f"{TRITON_UPSTREAM}/v2/health/ready",
            timeout=3,
        )
        ready = response.status_code == 200
    except httpx.HTTPError:
        pass
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "triton_ready": ready,
        "routes": {
            "web": "/",
            "health": "/health",
            "register": "/register",
            "speakers": "/speakers",
            "tts": "/tts/",
            "streaming_tts": "/tts/stream",
            "triton": "/v2/",
            "grpc_port": 18001,
            "metrics_port": 18002,
        },
    }


@app.api_route(
    "/v2/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def triton_proxy(path: str, request: Request):
    upstream_url = f"{TRITON_UPSTREAM}/v2/{path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    try:
        body = await request.body()
        upstream_request = request.app.state.http_client.build_request(
            request.method,
            upstream_url,
            headers=_forward_headers(request.headers),
            content=body,
        )
        upstream_response = await request.app.state.http_client.send(
            upstream_request,
            stream=True,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Triton upstream is unavailable",
                "detail": str(exc),
            },
        )

    response_headers = {
        name: value
        for name, value in upstream_response.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() != "content-length"
    }
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_response.aclose),
    )


app.mount(
    "/",
    StaticFiles(directory=WEB_DIR, html=True, check_dir=True),
    name="web",
)
