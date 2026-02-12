"""
TTS Routes - OpenAI-compatible Text-to-Speech API.

Implements:
- POST /v1/audio/speech
- GET /v1/audio/voices
"""

import os
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.config import settings
from app.models.tts_models import TTSRequest, MODEL_MAP, OPENAI_VOICE_MAP
from app.services.tts_service import TTSService


router = APIRouter(prefix="/v1/audio", tags=["Text-to-Speech"])

# Global service instances keyed by "mode:size" (e.g. "custom_voice:1.7b")
_tts_services: Dict[str, TTSService] = {}


def _service_key(mode: str, size: str) -> str:
    return f"{mode}:{size}"


def _get_model_path(mode: str, size: str) -> str:
    """Return the local model path for a given mode and size."""
    if mode == "custom_voice" and size == "1.7b":
        return settings.custom_voice_model_path
    elif mode == "custom_voice" and size == "0.6b":
        return settings.custom_voice_0_6b_model_path
    elif mode == "voice_design" and size == "1.7b":
        return settings.voice_design_model_path
    elif mode == "voice_clone" and size == "1.7b":
        return settings.voice_clone_model_path
    elif mode == "voice_clone" and size == "0.6b":
        return settings.voice_clone_0_6b_model_path
    else:
        raise ValueError(f"No model path for mode={mode}, size={size}")


def init_tts_services() -> Dict[str, TTSService]:
    """Initialize TTS service(s) on startup based on configuration."""
    global _tts_services

    model_type = settings.model_type.lower()
    tts_modes = settings.tts_modes.lower()

    active_modes = []
    if tts_modes == "all":
        active_modes = ["custom_voice", "voice_design", "voice_clone"]
    else:
        active_modes = [m.strip() for m in tts_modes.split(",")]

    active_sizes = []
    if model_type == "both":
        active_sizes = ["0.6b", "1.7b"]
    else:
        active_sizes = [model_type]

    for mode in active_modes:
        for size in active_sizes:
            # VoiceDesign only exists for 1.7B
            if mode == "voice_design" and size == "0.6b":
                print(f"[TTS] Skipping voice_design for 0.6B (not available)")
                continue

            key = _service_key(mode, size)
            try:
                path = _get_model_path(mode, size)
            except ValueError as e:
                print(f"[TTS] Skipping {key}: {e}")
                continue

            if not os.path.isdir(path):
                print(f"[TTS] Warning: Model path not found: {path}, skipping {key}")
                continue

            print(f"[TTS] Initializing {key} from {path}...")
            service = TTSService(model_path=path, model_size=size, mode=mode)
            service.load()
            _tts_services[key] = service

    return _tts_services


def get_tts_service(model_name: str) -> TTSService:
    """Resolve model name to the appropriate TTSService instance."""
    if not _tts_services:
        raise HTTPException(status_code=503, detail="TTS service not initialized")

    # Lookup mode and size from model name
    model_lower = model_name.lower()
    if model_lower in MODEL_MAP:
        mode, size = MODEL_MAP[model_lower]
    else:
        # Default fallback
        mode, size = "custom_voice", settings.model_type.lower()
        if size == "both":
            size = "1.7b"

    key = _service_key(mode, size)

    if key in _tts_services:
        service = _tts_services[key]
        if not service.is_loaded:
            raise HTTPException(status_code=503, detail=f"TTS model {key} not loaded")
        return service

    # Try alternate size for the same mode only
    alt_size = "1.7b" if size == "0.6b" else "0.6b"
    alt_key = _service_key(mode, alt_size)
    if alt_key in _tts_services:
        service = _tts_services[alt_key]
        if service.is_loaded:
            return service

    # For custom_voice mode with unknown model names, fallback to any custom_voice service
    if mode == "custom_voice":
        for svc in _tts_services.values():
            if svc.is_loaded and svc.mode == "custom_voice":
                return svc

    # No cross-mode fallback: return a clear error
    loaded_modes = sorted({svc.mode for svc in _tts_services.values() if svc.is_loaded})
    raise HTTPException(
        status_code=400,
        detail=f"Model '{model_name}' requires mode '{mode}', but it is not loaded. "
               f"Loaded modes: {loaded_modes}",
    )


# Media type mapping
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@router.post("/speech")
async def create_speech(request: TTSRequest):
    """Generate speech from text (OpenAI-compatible)."""
    # Validate input
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    if len(request.input) > settings.max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input text too long. Maximum {settings.max_input_length} characters.",
        )

    if request.response_format not in MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {list(MEDIA_TYPES.keys())}",
        )

    # Resolve model → service
    service = get_tts_service(request.model)

    # Mode-specific validation
    if service.mode == "voice_design" and not request.instruct:
        raise HTTPException(
            status_code=400,
            detail="VoiceDesign mode requires the 'instruct' field with a voice description.",
        )
    if service.mode == "voice_clone" and not request.ref_audio:
        raise HTTPException(
            status_code=400,
            detail="VoiceClone mode requires the 'ref_audio' field with base64-encoded reference audio.",
        )

    try:
        audio, sr = service.synthesize(request)
        audio_bytes = TTSService.convert_audio(audio, sr, request.response_format)

        media_type = MEDIA_TYPES.get(request.response_format, "audio/wav")

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")


@router.get("/voices")
async def list_voices() -> dict:
    """List available voices from all loaded TTS services."""
    if not _tts_services:
        raise HTTPException(status_code=503, detail="TTS service not initialized")

    all_voices = {}
    for key, service in _tts_services.items():
        if not service.is_loaded:
            continue

        if service.mode in ("custom_voice",):
            speakers = service.speakers or []
            for spk in speakers:
                if spk not in all_voices:
                    all_voices[spk] = {
                        "voice_id": spk,
                        "name": spk,
                        "mode": "custom_voice",
                        "models": [service.model_id],
                    }
                else:
                    if service.model_id not in all_voices[spk]["models"]:
                        all_voices[spk]["models"].append(service.model_id)

        elif service.mode == "voice_clone":
            if "custom" not in all_voices:
                all_voices["custom"] = {
                    "voice_id": "custom",
                    "name": "Custom (via ref_audio)",
                    "mode": "voice_clone",
                    "models": [service.model_id],
                }
            else:
                if service.model_id not in all_voices["custom"]["models"]:
                    all_voices["custom"]["models"].append(service.model_id)

        elif service.mode == "voice_design":
            if "designed" not in all_voices:
                all_voices["designed"] = {
                    "voice_id": "designed",
                    "name": "Designed (via instruct)",
                    "mode": "voice_design",
                    "models": [service.model_id],
                }
            else:
                if service.model_id not in all_voices["designed"]["models"]:
                    all_voices["designed"]["models"].append(service.model_id)

    # Add OpenAI voice aliases
    openai_aliases = []
    for alias, speaker in OPENAI_VOICE_MAP.items():
        openai_aliases.append({
            "voice_id": alias,
            "name": f"{alias} (→ {speaker})",
            "mode": "custom_voice",
            "models": [],
        })

    return {
        "voices": list(all_voices.values()),
        "openai_aliases": openai_aliases,
        "default_voice": settings.default_speaker,
    }
