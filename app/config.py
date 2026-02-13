"""Configuration management for Qwen3-TTS API Server."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # 1.7B model paths
    custom_voice_model_path: str = "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    voice_design_model_path: str = "/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    voice_clone_model_path: str = "/models/Qwen3-TTS-12Hz-1.7B-Base"

    # 0.6B model paths
    custom_voice_0_6b_model_path: str = "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    voice_clone_0_6b_model_path: str = "/models/Qwen3-TTS-12Hz-0.6B-Base"

    # Model selection
    model_type: str = "1.7b"  # "0.6b", "1.7b", "both"
    tts_modes: str = "custom_voice,voice_clone"  # "custom_voice", "voice_design", "voice_clone", "all"

    # Reference audio directory for voice clone
    ref_audio_dir: str = "/models/ref_audio"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Defaults
    default_speaker: str = "Vivian"
    default_language: str = "Auto"
    max_input_length: int = 4096

    # Generation parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048

    # Audio
    sample_rate: int = 24000

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)
