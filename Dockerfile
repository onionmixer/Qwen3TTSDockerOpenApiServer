FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Qwen3-TTS from local source
COPY Qwen3-TTS/ /opt/Qwen3-TTS/
RUN pip install --no-cache-dir -e /opt/Qwen3-TTS/

# Fix torchvision/torchaudio to match upgraded torch version
RUN pip install --no-cache-dir --force-reinstall torchvision torchaudio

# Install Flash Attention
RUN MAX_JOBS=4 pip install -U flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed, will use SDPA"

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix numba compatibility with numpy>=2.x (must be last to avoid downgrade)
RUN pip install --no-cache-dir -U "numba>=0.61" "setuptools<82"

# Copy application code
COPY app/ /workspace/app/

# Environment defaults
ENV DEVICE=cuda
ENV DTYPE=bfloat16
ENV ATTN_IMPLEMENTATION=flash_attention_2
ENV MODEL_TYPE=1.7b
ENV TTS_MODES=all
ENV CUSTOM_VOICE_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
ENV VOICE_DESIGN_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
ENV VOICE_CLONE_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base
ENV CUSTOM_VOICE_0_6B_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
ENV VOICE_CLONE_0_6B_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-Base
ENV REF_AUDIO_DIR=/models/ref_audio
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV DEFAULT_SPEAKER=Vivian
ENV DEFAULT_LANGUAGE=Auto

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
