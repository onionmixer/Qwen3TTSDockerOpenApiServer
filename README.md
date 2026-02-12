# Qwen3-TTS Docker OpenAI API Server

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba)를 OpenAI TTS API 규격으로 서빙하는 Docker 프로젝트입니다.

OpenAI의 `/v1/audio/speech` API를 사용하는 클라이언트(Open WebUI, ChatBox 등)에서 Qwen3-TTS를 **드롭인 대체(drop-in replacement)**로 사용할 수 있습니다.

## 원본 프로젝트

- **Qwen3-TTS** : https://github.com/QwenLM/Qwen3-TTS
- Alibaba Qwen 팀이 개발한 오픈소스 Text-to-Speech 모델
- 12Hz 코덱 기반, CustomVoice / VoiceDesign / VoiceClone 3가지 모드 지원
- 0.6B (경량) 및 1.7B (고품질) 두 가지 모델 사이즈 제공

본 프로젝트는 Qwen3-TTS의 추론 API를 래핑하여 OpenAI 호환 REST API로 노출하는 **서빙 레이어**입니다. 모델 자체의 학습이나 수정은 포함하지 않습니다.

## 지원 기능

| 기능 | 설명 |
|------|------|
| **CustomVoice** | 내장 스피커(Vivian, Ryan, Serena 등 9종)로 TTS |
| **VoiceDesign** | 자연어 지시문으로 새로운 음성 스타일 생성 (1.7B만) |
| **VoiceClone** | 참조 오디오로 음성 복제 |
| **듀얼 모델** | 0.6B / 1.7B 동시 로드 가능 |
| **OpenAI 호환** | `tts-1`, `tts-1-hd` 모델명 및 `alloy`~`shimmer` 음성 매핑 |
| **다중 포맷** | WAV, MP3, Opus, AAC, FLAC, PCM 출력 |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/v1/audio/speech` | 텍스트를 음성으로 변환 (OpenAI 호환) |
| `GET` | `/v1/audio/voices` | 사용 가능한 음성 목록 |
| `GET` | `/v1/models` | 로드된 모델 목록 (OpenAI 호환) |
| `GET` | `/health` | 서비스 상태 확인 |
| `GET` | `/docs` | Swagger UI |

## 모델 이름 매핑

| model 값 | 모드 | 사이즈 |
|-----------|------|--------|
| `qwen3-tts-1.7b` | CustomVoice | 1.7B |
| `qwen3-tts-0.6b` | CustomVoice | 0.6B |
| `qwen3-tts-1.7b-design` | VoiceDesign | 1.7B |
| `qwen3-tts-1.7b-clone` | VoiceClone | 1.7B |
| `qwen3-tts-0.6b-clone` | VoiceClone | 0.6B |
| `tts-1` | CustomVoice | 0.6B |
| `tts-1-hd` | CustomVoice | 1.7B |

## 빠른 시작

### 1. 모델 다운로드

HuggingFace에서 필요한 모델을 다운로드합니다. 사용할 모드에 따라 필요한 모델만 받아도 됩니다.

```bash
pip install huggingface_hub
```

**전체 모델 일괄 다운로드** (~18GB):

```bash
python -c "
from huggingface_hub import snapshot_download
for repo in [
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
    'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
    'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
    'Qwen/Qwen3-TTS-Tokenizer-12Hz',
]:
    name = repo.split('/')[-1]
    snapshot_download(repo, local_dir=f'models/{name}')
    print(f'Done: {name}')
"
```

**모드별 필요 모델:**

| 모드 | 필요 모델 | 용량 |
|------|-----------|------|
| CustomVoice (1.7B) | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~4.3GB |
| CustomVoice (0.6B) | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~2.4GB |
| VoiceDesign (1.7B) | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~4.3GB |
| VoiceClone (1.7B) | `Qwen3-TTS-12Hz-1.7B-Base` | ~4.3GB |
| VoiceClone (0.6B) | `Qwen3-TTS-12Hz-0.6B-Base` | ~2.4GB |

> Tokenizer (`Qwen3-TTS-Tokenizer-12Hz`, ~651MB)는 모든 모드에서 사용되며, 각 모델 디렉토리에 포함되어 있거나 자동으로 참조됩니다.

**개별 다운로드 예시** (CustomVoice 1.7B만):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

다운로드 후 디렉토리 구조는 다음과 같아야 합니다:

```
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── ...
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
├── Qwen3-TTS-12Hz-0.6B-Base/
└── Qwen3-TTS-Tokenizer-12Hz/
```

### 2. Docker Compose 실행

#### 2-1. 환경 설정

```bash
cp .env.example .env
```

`.env` 파일에서 `MODEL_DIR`을 모델이 저장된 **절대 경로**로 설정합니다:

```env
# 모델 다운로드 경로 (필수)
MODEL_DIR=/home/user/models

# 사용할 모델 사이즈: 0.6b, 1.7b, both
MODEL_TYPE=1.7b

# 활성화할 TTS 모드: custom_voice, voice_design, voice_clone, all
TTS_MODES=all

# 서버 포트
API_PORT=8080
```

`MODEL_DIR` 아래에 다운로드한 모델 디렉토리들이 있어야 합니다. Docker Compose에서 이 경로를 컨테이너 내부의 `/models`로 읽기 전용 마운트합니다:

```
호스트: ${MODEL_DIR}/Qwen3-TTS-12Hz-1.7B-CustomVoice/
   ↓ volume mount
컨테이너: /models/Qwen3-TTS-12Hz-1.7B-CustomVoice/
```

#### 2-2. 빌드 및 실행

```bash
docker compose up --build
```

> **참고:** Dockerfile에서 [Qwen3-TTS 소스코드](https://github.com/QwenLM/Qwen3-TTS)를 빌드합니다. 프로젝트 루트에 `Qwen3-TTS/` 디렉토리가 필요합니다:
> ```bash
> git clone https://github.com/QwenLM/Qwen3-TTS.git
> docker compose up --build
> ```

#### 2-3. VoiceClone 참조 오디오 마운트 (선택)

VoiceClone 모드에서 파일 기반 참조 오디오를 사용하려면 `REF_AUDIO_DIR`도 설정합니다:

```env
REF_AUDIO_DIR=/home/user/ref_audio
```

이 디렉토리는 컨테이너 내부 `/models/ref_audio`로 마운트됩니다.

### 3. 로컬 실행 (Docker 없이)

```bash
# Qwen3-TTS 패키지 설치
pip install /path/to/Qwen3-TTS

# 서버 의존성 설치
pip install -r requirements.txt

# 환경변수 설정 후 실행
export CUSTOM_VOICE_MODEL_PATH=./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
export MODEL_TYPE=1.7b
export TTS_MODES=custom_voice

python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 사용 예시

### CustomVoice

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b","input":"안녕하세요","voice":"Vivian"}' \
  --output output.wav
```

### OpenAI 호환 요청

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1-hd","input":"Hello world","voice":"alloy"}' \
  --output output.wav
```

### VoiceDesign

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b-design","input":"Hello!","instruct":"A warm friendly female voice"}' \
  --output output.wav
```

### VoiceClone

```bash
REF_B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-tts-1.7b-clone\",\"input\":\"Hello\",\"ref_audio\":\"${REF_B64}\",\"ref_text\":\"Reference text\"}" \
  --output output.wav
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_TYPE` | `1.7b` | 로드할 모델 사이즈 (`0.6b`, `1.7b`, `both`) |
| `TTS_MODES` | `custom_voice` | 활성화할 모드 (`custom_voice`, `voice_design`, `voice_clone`, `all`) |
| `DEVICE` | `cuda` | PyTorch 디바이스 |
| `DTYPE` | `bfloat16` | 모델 정밀도 |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention 구현 (fallback: `sdpa`) |
| `DEFAULT_SPEAKER` | `Vivian` | 기본 스피커 |
| `API_PORT` | `8080` | 서버 포트 |

전체 목록은 [.env.example](.env.example) 참조.

## GPU 호환성

Qwen3-TTS가 의존하는 `transformers==4.57.3`은 `torch>=2.10`을 요구하며, PyTorch 2.10은 **CUDA Compute Capability 7.0 (Volta) 이상**만 지원합니다.

### 지원 GPU

| GPU 세대 | Compute Capability | 지원 여부 |
|----------|-------------------|-----------|
| Pascal (GTX 1080 Ti, TITAN Xp 등) | sm_61 | **미지원** |
| Volta (V100) | sm_70 | 지원 |
| Turing (RTX 2060~2080 Ti) | sm_75 | 지원 |
| Ampere (RTX 3060~3090, A100) | sm_80, sm_86 | 지원 |
| Ada Lovelace (RTX 4060~4090) | sm_89 | 지원 |
| Hopper (H100) | sm_90 | 지원 |

### VRAM 요구사항

| 모델 | dtype | 예상 VRAM | 권장 GPU |
|------|-------|-----------|----------|
| 0.6B | bfloat16 | ~2GB | 8GB+ |
| 1.7B | bfloat16 | ~4GB | 8GB+ |
| 전체 (both + all) | bfloat16 | ~18GB | 24GB+ |

### Pascal GPU (GTX 1080 Ti 등) 사용 시

Pascal 세대 GPU에서는 CUDA 모드를 사용할 수 없습니다. 다음과 같이 **CPU 모드**로 실행할 수 있습니다:

```env
DEVICE=cpu
DTYPE=float32
ATTN_IMPLEMENTATION=sdpa
```

> **참고:** CPU 모드에서는 0.6B 모델 기준 문장당 약 10~15초가 소요됩니다. 1.7B 모델은 더 오래 걸릴 수 있으므로 0.6B 사용을 권장합니다.

docker-compose.yml에서 CPU 모드로 실행하려면 `runtime: nvidia`와 `deploy.resources` 블록을 제거하고 환경변수를 수정합니다:

```yaml
services:
  qwen3-tts-api:
    build:
      context: .
    # runtime: nvidia  # 제거
    environment:
      - DEVICE=cpu
      - DTYPE=float32
      - ATTN_IMPLEMENTATION=sdpa
      - MODEL_TYPE=0.6b
      - TTS_MODES=custom_voice
      # ... 기타 환경변수
    volumes:
      - ${MODEL_DIR}:/models:ro
    ports:
      - "${API_PORT:-8080}:8080"
    # deploy 블록 제거
```

## 프로젝트 구조

```
app/
├── __init__.py              # 버전 정보
├── config.py                # pydantic-settings 환경변수 설정
├── main.py                  # FastAPI 앱, lifespan, 미들웨어
├── routes/
│   └── tts.py               # POST /v1/audio/speech, GET /v1/audio/voices
├── services/
│   └── tts_service.py       # Qwen3TTSModel 래핑, 추론, 오디오 변환
└── models/
    └── tts_models.py        # TTSRequest, VoiceInfo Pydantic 스키마
```

## 라이선스

이 서빙 레이어는 원본 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)의 Apache 2.0 라이선스를 따릅니다.
