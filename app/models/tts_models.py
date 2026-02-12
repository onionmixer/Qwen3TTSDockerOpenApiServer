"""TTS API Pydantic models - OpenAI compatible with Qwen3-TTS extensions."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# Model name → (mode, size) mapping
MODEL_MAP = {
    "qwen3-tts-1.7b": ("custom_voice", "1.7b"),
    "qwen3-tts-0.6b": ("custom_voice", "0.6b"),
    "qwen3-tts-1.7b-design": ("voice_design", "1.7b"),
    "qwen3-tts-1.7b-clone": ("voice_clone", "1.7b"),
    "qwen3-tts-0.6b-clone": ("voice_clone", "0.6b"),
    "tts-1": ("custom_voice", "0.6b"),
    "tts-1-hd": ("custom_voice", "1.7b"),
}

# OpenAI voice → Qwen3 speaker mapping
OPENAI_VOICE_MAP = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "onyx": "Uncle_Fu",
    "nova": "Ono_Anna",
    "shimmer": "Sohee",
}


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request with Qwen3-TTS extensions."""
    model: str = Field(
        default="qwen3-tts-1.7b",
        description="Model to use for TTS.",
    )
    input: str = Field(
        ...,
        description="The text to synthesize into speech.",
        max_length=4096,
    )
    voice: str = Field(
        default="Vivian",
        description="Speaker name for CustomVoice/VoiceClone modes.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="wav",
        description="Audio output format.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed (accepted for OpenAI compatibility, not applied).",
    )
    # Qwen3-TTS extensions
    language: Optional[str] = Field(
        default=None,
        description="Language for synthesis (None → Auto).",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="CustomVoice: style instruction / VoiceDesign: voice description.",
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="VoiceClone: base64-encoded reference audio.",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="VoiceClone: transcript of reference audio.",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model": "qwen3-tts-1.7b",
                    "input": "Hello, how are you today?",
                    "voice": "Vivian",
                    "response_format": "wav",
                },
                {
                    "model": "tts-1-hd",
                    "input": "Hello world",
                    "voice": "alloy",
                    "response_format": "mp3",
                },
            ]
        }


class VoiceInfo(BaseModel):
    """Voice information."""
    voice_id: str
    name: str
    mode: str
    models: list[str] = []
