"""TTS Service - Qwen3-TTS wrapper for OpenAI-compatible API."""

import io
import subprocess
from typing import Optional, List, Tuple

import numpy as np
import torch

from app.config import settings, get_torch_dtype
from app.models.tts_models import TTSRequest, OPENAI_VOICE_MAP


class TTSService:
    """
    Text-to-Speech service wrapping Qwen3TTSModel.

    Each instance handles one (mode, size) combination:
    - custom_voice + 1.7b/0.6b
    - voice_design + 1.7b
    - voice_clone + 1.7b/0.6b
    """

    def __init__(self, model_path: str, model_size: str, mode: str):
        self.model_path = model_path
        self.model_size = model_size
        self.mode = mode
        self.model = None
        self._loaded = False
        self.speakers: Optional[List[str]] = None
        self.languages: Optional[List[str]] = None
        self.sample_rate = settings.sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_id(self) -> str:
        if self.mode == "custom_voice":
            return f"qwen3-tts-{self.model_size}"
        elif self.mode == "voice_design":
            return f"qwen3-tts-{self.model_size}-design"
        else:
            return f"qwen3-tts-{self.model_size}-clone"

    def load(self) -> None:
        """Load the Qwen3-TTS model."""
        if self._loaded:
            return

        from qwen_tts import Qwen3TTSModel

        device = settings.device
        dtype = get_torch_dtype(settings.dtype)
        attn_impl = settings.attn_implementation

        print(f"[TTS-{self.mode}-{self.model_size}] Loading from {self.model_path}")
        print(f"[TTS-{self.mode}-{self.model_size}] Device: {device}, dtype: {dtype}, attn: {attn_impl}")

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            if "flash" in attn_impl.lower():
                print(f"[TTS-{self.mode}-{self.model_size}] Flash attention failed ({e}), falling back to SDPA")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    device_map=device,
                    dtype=dtype,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.speakers = self.model.get_supported_speakers()
        self.languages = self.model.get_supported_languages()

        self._loaded = True
        print(f"[TTS-{self.mode}-{self.model_size}] Model loaded successfully")
        if self.speakers:
            print(f"[TTS-{self.mode}-{self.model_size}] Speakers: {self.speakers}")

    def _resolve_speaker(self, voice: str) -> str:
        """Resolve voice name to a valid Qwen3 speaker."""
        if self.speakers is None:
            return voice

        speakers_lower = {s.lower(): s for s in self.speakers}

        # Direct match (case-insensitive against lowered speaker list)
        if voice.lower() in speakers_lower:
            return speakers_lower[voice.lower()]

        # OpenAI voice mapping
        if voice.lower() in OPENAI_VOICE_MAP:
            mapped = OPENAI_VOICE_MAP[voice.lower()]
            if mapped.lower() in speakers_lower:
                return speakers_lower[mapped.lower()]

        # Fallback to default speaker
        default = settings.default_speaker
        if default.lower() in speakers_lower:
            return speakers_lower[default.lower()]

        # Last resort: first available speaker
        if self.speakers:
            return self.speakers[0]

        return voice

    def synthesize(self, request: TTSRequest) -> Tuple[np.ndarray, int]:
        """Run TTS inference based on mode."""
        if not self._loaded:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        language = request.language or settings.default_language
        gen_kwargs = dict(
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            max_new_tokens=settings.max_new_tokens,
        )

        if self.mode == "custom_voice":
            speaker = self._resolve_speaker(request.voice)
            wavs, sr = self.model.generate_custom_voice(
                text=request.input,
                speaker=speaker,
                language=language,
                instruct=request.instruct,
                **gen_kwargs,
            )
        elif self.mode == "voice_design":
            wavs, sr = self.model.generate_voice_design(
                text=request.input,
                instruct=request.instruct or "",
                language=language,
                **gen_kwargs,
            )
        elif self.mode == "voice_clone":
            wavs, sr = self.model.generate_voice_clone(
                text=request.input,
                language=language,
                ref_audio=request.ref_audio,
                ref_text=request.ref_text,
                x_vector_only_mode=(request.ref_text is None or request.ref_text == ""),
                **gen_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return wavs[0], sr

    @staticmethod
    def convert_audio(audio: np.ndarray, sr: int, fmt: str) -> bytes:
        """Convert audio numpy array to the requested format."""
        import scipy.io.wavfile as wav_io

        # Normalize
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        # Create WAV bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        wav_io.write(wav_buffer, sr, audio_int16)
        wav_bytes = wav_buffer.getvalue()

        if fmt == "wav":
            return wav_bytes

        if fmt == "pcm":
            return audio_int16.tobytes()

        if fmt in ("mp3", "opus", "aac", "flac"):
            return TTSService._convert_with_ffmpeg(wav_bytes, sr, fmt)

        return wav_bytes

    @staticmethod
    def _convert_with_ffmpeg(wav_bytes: bytes, sr: int, fmt: str) -> bytes:
        """Convert audio using ffmpeg."""
        format_args = {
            "mp3": ["-f", "mp3", "-acodec", "libmp3lame", "-ab", "192k"],
            "opus": ["-f", "opus", "-acodec", "libopus", "-ab", "128k"],
            "aac": ["-f", "adts", "-acodec", "aac", "-ab", "192k"],
            "flac": ["-f", "flac", "-acodec", "flac"],
        }
        args = format_args.get(fmt, ["-f", "wav"])

        cmd = [
            "ffmpeg",
            "-f", "wav",
            "-i", "pipe:0",
            *args,
            "-ar", str(sr),
            "-ac", "1",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                input=wav_bytes,
                capture_output=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[TTS] ffmpeg conversion failed: {e.stderr.decode()}")
            return wav_bytes
        except FileNotFoundError:
            print("[TTS] ffmpeg not found, returning WAV")
            return wav_bytes
