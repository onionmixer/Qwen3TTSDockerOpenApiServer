"""
Qwen3-TTS OpenAI-Compatible API Server

Main FastAPI application providing OpenAI-compatible TTS API.
"""

import gc
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.routes import tts_router
from app.routes.tts import init_tts_services, _tts_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    print("=" * 60)
    print(f"Qwen3-TTS OpenAI API Server v{__version__}")
    print("=" * 60)
    print(f"Device: {settings.device}")
    print(f"Dtype: {settings.dtype}")
    print(f"Attention: {settings.attn_implementation}")
    print(f"Model Type: {settings.model_type}")
    print(f"TTS Modes: {settings.tts_modes}")
    print("=" * 60)

    print("\n[Startup] Initializing TTS services...")
    try:
        init_tts_services()
        print("[Startup] TTS services ready")
    except Exception as e:
        print(f"[Startup] TTS service initialization failed: {e}")

    print("\n" + "=" * 60)
    print(f"Server ready at http://{settings.api_host}:{settings.api_port}")
    print("=" * 60 + "\n")

    yield

    print("\n[Shutdown] Cleaning up...")
    for key, service in list(_tts_services.items()):
        print(f"[Shutdown] Unloading {key}...")
        if service.model is not None:
            del service.model
            service.model = None
            service._loaded = False
    _tts_services.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] GPU memory released")


app = FastAPI(
    title="Qwen3-TTS OpenAI API Server",
    description="OpenAI-compatible TTS API using Qwen3-TTS models",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Include TTS router
app.include_router(tts_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Qwen3-TTS OpenAI API Server",
        "version": __version__,
        "endpoints": {
            "tts": "/v1/audio/speech",
            "voices": "/v1/audio/voices",
            "models": "/v1/models",
            "docs": "/docs",
        },
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    tts_status = {}
    if not _tts_services:
        tts_status = {"status": "no_models_loaded"}
    else:
        for key, service in _tts_services.items():
            tts_status[key] = "ready" if service.is_loaded else "loading"

    all_ready = all(s == "ready" for s in tts_status.values()) if tts_status else False

    return {
        "status": "healthy" if all_ready else "degraded",
        "services": {
            "tts": tts_status,
        },
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = []

    if _tts_services:
        seen_ids = set()
        for key, service in _tts_services.items():
            mid = service.model_id
            if mid not in seen_ids:
                seen_ids.add(mid)
                models.append({
                    "id": mid,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "alibaba",
                    "permission": [],
                    "root": mid,
                    "parent": None,
                })

    # Always advertise OpenAI-compatible aliases
    for alias_id in ("tts-1", "tts-1-hd"):
        models.append({
            "id": alias_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "alibaba",
            "permission": [],
            "root": alias_id,
            "parent": None,
        })

    return {
        "object": "list",
        "data": models,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler (OpenAI error format)."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "code": "internal_error",
            }
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
