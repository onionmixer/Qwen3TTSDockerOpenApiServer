# Qwen3-TTS OpenAI-Compatible API Server 개발 계획

## Context

Qwen3-TTS(Alibaba의 오픈소스 TTS 모델)를 활용하여, 기존 VibeVoiceDockerOpenaiApiServer와 동일한 패턴의 OpenAI 호환 TTS API 서버를 구축합니다. 이를 통해 OpenAI TTS API를 사용하는 클라이언트(Open WebUI 등)에서 Qwen3-TTS를 드롭인 대체(drop-in replacement)로 사용할 수 있게 합니다.

**핵심 요구사항:**
- 3가지 TTS 모드 모두 지원: CustomVoice, VoiceDesign, VoiceClone
- 0.6B + 1.7B 듀얼 모델 지원
- 로컬 모델 경로 전용 (Docker volume mount)
- STT 없이 TTS만

**지원 언어 (10개 + Auto):**
- Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- `Auto`: 언어 자동 감지 모드

**0.6B 모델 제한사항:**
- 0.6B CustomVoice: `instruct`(스타일 지시) 미지원 → 자동으로 `None` 처리
- 0.6B VoiceDesign: 모델 자체 없음 → 요청 시 HTTP 400 반환
- 0.6B Base(VoiceClone): 지원

---

## 프로젝트 구조

```
/mnt/USERS/onion/DATA_ORIGN/Workspace/Qwen3TTSDockerOpenApiServer/
├── app/
│   ├── __init__.py              # version = "1.0.0"
│   ├── main.py                  # FastAPI app, lifespan, middleware, error handlers
│   ├── config.py                # pydantic-settings 환경변수 설정
│   ├── routes/
│   │   ├── __init__.py          # tts_router export
│   │   └── tts.py               # POST /v1/audio/speech, GET /v1/audio/voices
│   ├── services/
│   │   ├── __init__.py          # TTSService export
│   │   └── tts_service.py       # 모델 로딩, 추론, 오디오 변환
│   └── models/
│       ├── __init__.py          # Pydantic 모델 export
│       └── tts_models.py        # TTSRequest, VoiceInfo 등 스키마
├── Dockerfile
├── docker-compose.yml
├── docker-compose.with-openwebui.yml  # Open WebUI 연동 배포
├── requirements.txt
├── .env.example
├── .gitignore
└── run-test.sh                  # 자동화 빌드/테스트 스크립트
```

---

## 1단계: 설정 및 모델 정의 (`app/config.py`, `app/models/tts_models.py`)

### `app/config.py` — 환경변수 설정

pydantic-settings `BaseSettings` 사용. 주요 설정:

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `DEVICE` | `cuda` | PyTorch 디바이스 |
| `DTYPE` | `bfloat16` | 모델 정밀도 (`bfloat16`, `float16`) |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention 구현 (fallback: `sdpa`) |
| `CUSTOM_VOICE_MODEL_PATH` | `/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B CustomVoice 모델 경로 |
| `VOICE_DESIGN_MODEL_PATH` | `/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B VoiceDesign 모델 경로 |
| `VOICE_CLONE_MODEL_PATH` | `/models/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B VoiceClone(Base) 모델 경로 |
| `CUSTOM_VOICE_0_6B_MODEL_PATH` | `/models/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B CustomVoice 모델 경로 |
| `VOICE_CLONE_0_6B_MODEL_PATH` | `/models/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B VoiceClone(Base) 모델 경로 |
| `MODEL_TYPE` | `1.7b` | 로드할 모델: `0.6b`, `1.7b`, `both` |
| `TTS_MODES` | `custom_voice` | 활성화할 모드: `custom_voice`, `voice_design`, `voice_clone`, `all` |
| `REF_AUDIO_DIR` | `/models/ref_audio` | Voice clone 참조 오디오 디렉토리 |
| `API_HOST` | `0.0.0.0` | 서버 바인드 주소 |
| `API_PORT` | `8080` | 서버 포트 |
| `DEFAULT_SPEAKER` | `Vivian` | 기본 스피커 |
| `DEFAULT_LANGUAGE` | `Auto` | 기본 언어 |
| `MAX_INPUT_LENGTH` | `4096` | 입력 텍스트 최대 길이 |
| `TEMPERATURE` | `0.9` | 생성 temperature |
| `TOP_K` | `50` | Top-k 샘플링 |
| `TOP_P` | `1.0` | Top-p 샘플링 |
| `REPETITION_PENALTY` | `1.05` | 반복 패널티 |
| `MAX_NEW_TOKENS` | `2048` | 최대 생성 토큰 수 |
| `SUBTALKER_DOSAMPLE` | `True` | Sub-codebook 샘플링 여부 |
| `SUBTALKER_TOP_K` | `50` | Sub-codebook Top-k |
| `SUBTALKER_TOP_P` | `1.0` | Sub-codebook Top-p |
| `SUBTALKER_TEMPERATURE` | `0.9` | Sub-codebook Temperature |
| `SAMPLE_RATE` | `24000` | 오디오 샘플레이트 |
| `LOG_LEVEL` | `INFO` | 로깅 레벨 |

### `app/models/tts_models.py` — API 스키마

```python
class TTSRequest(BaseModel):
    model: str = "qwen3-tts-1.7b"          # 모델 선택 (아래 매핑 참조)
    input: str                              # 합성할 텍스트 (필수)
    voice: str = "Vivian"                   # 스피커 이름 (CustomVoice/VoiceClone)
    response_format: Literal["mp3","opus","aac","flac","wav","pcm"] = "wav"
    speed: float = 1.0                      # OpenAI 호환 (수용만, 미구현)
    # --- Qwen3-TTS 확장 필드 ---
    language: Optional[str] = None          # 언어 (None → Auto)
    instruct: Optional[str] = None          # CustomVoice: 스타일 지시 / VoiceDesign: 음성 설명
    ref_audio: Optional[str] = None         # VoiceClone: base64 인코딩 참조 오디오 또는 프리셋 이름
    ref_text: Optional[str] = None          # VoiceClone: 참조 오디오 텍스트 (ICL 모드용)
    x_vector_only: Optional[bool] = False   # VoiceClone: x-vector only 모드 (ref_text 불필요)
```

**모델 이름 → 모드/사이즈 매핑:**

| model 값 | 모드 | 사이즈 | 설명 |
|-----------|------|--------|------|
| `qwen3-tts-1.7b` | CustomVoice | 1.7B | 기본값, 고품질 |
| `qwen3-tts-0.6b` | CustomVoice | 0.6B | 경량 (instruct 미지원) |
| `qwen3-tts-1.7b-design` | VoiceDesign | 1.7B | 음성 디자인 |
| `qwen3-tts-1.7b-clone` | VoiceClone | 1.7B | 음성 클론 |
| `qwen3-tts-0.6b-clone` | VoiceClone | 0.6B | 경량 음성 클론 |
| `tts-1` | CustomVoice | 0.6B | OpenAI 호환 |
| `tts-1-hd` | CustomVoice | 1.7B | OpenAI 호환 |

**OpenAI 표준 voice → Qwen3 speaker 매핑:**

| OpenAI voice | Qwen3 speaker |
|-------------|---------------|
| `alloy` | `Vivian` |
| `echo` | `Ryan` |
| `fable` | `Serena` |
| `onyx` | `Uncle_Fu` |
| `nova` | `Ono_Anna` |
| `shimmer` | `Sohee` |

---

## 2단계: TTS 서비스 (`app/services/tts_service.py`)

핵심 추론 계층. `Qwen3TTSModel`을 래핑하여 3가지 모드를 처리합니다.

### 클래스 설계

```python
class TTSService:
    def __init__(self, model_path: str, model_type: str, mode: str):
        """
        model_path: 로컬 모델 경로
        model_type: "0.6b" | "1.7b"
        mode: "custom_voice" | "voice_design" | "voice_clone"
        """
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=f"cuda:{device}",
            dtype=torch_dtype,
            attn_implementation=attn_impl,  # flash_attention_2, fallback to sdpa
        )
        self.mode = mode
        self.model_type = model_type
        self.speakers = self.model.get_supported_speakers()
        self.languages = self.model.get_supported_languages()
        self._inference_lock = asyncio.Lock()  # 동시성 제어

    async def synthesize(self, request: TTSRequest) -> Tuple[np.ndarray, int]:
        """모드에 따라 적절한 generate 메서드 호출 (Lock으로 동시성 보호)"""
        async with self._inference_lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._synthesize_sync, request
            )

    def _synthesize_sync(self, request: TTSRequest) -> Tuple[np.ndarray, int]:
        if self.mode == "custom_voice":
            # 0.6B일 때 instruct 자동 None 처리
            instruct = None if self.model_type == "0.6b" else request.instruct
            wavs, sr = self.model.generate_custom_voice(
                text=request.input,
                speaker=resolved_speaker,
                language=request.language or "Auto",
                instruct=instruct,
            )
        elif self.mode == "voice_design":
            wavs, sr = self.model.generate_voice_design(
                text=request.input,
                instruct=request.instruct,  # 필수
                language=request.language or "Auto",
            )
        elif self.mode == "voice_clone":
            # base64 → temp file (with cleanup)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(decoded_audio_bytes)
                tmp.flush()
                wavs, sr = self.model.generate_voice_clone(
                    text=request.input,
                    language=request.language or "Auto",
                    ref_audio=tmp.name,
                    ref_text=request.ref_text,
                    x_vector_only_mode=request.x_vector_only,
                )
        return self._normalize_audio(wavs[0]), sr

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        """피크를 [-1, 1] 범위로 정규화"""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    @staticmethod
    def convert_audio(audio: np.ndarray, sr: int, fmt: str) -> bytes:
        """WAV/MP3/Opus/AAC/FLAC/PCM 변환 (ffmpeg 사용)"""
```

### 모듈 레벨 서비스 관리

```python
# tts.py (routes) 에서 관리
_tts_services: Dict[str, TTSService] = {}

def init_tts_services():
    """설정에 따라 필요한 모델들을 로드"""
    # MODEL_TYPE에 따라 0.6b, 1.7b, both
    # TTS_MODES에 따라 custom_voice, voice_design, voice_clone, all

def get_tts_service(model_name: str) -> TTSService:
    """모델 이름을 파싱하여 적절한 서비스 반환"""
```

**핵심 로직:**
- `MODEL_TYPE=both` + `TTS_MODES=all` → 최대 5개 모델 인스턴스 (0.6B에는 VoiceDesign 없음)
- Attention 구현 fallback: `flash_attention_2` 시도 → 실패시 `sdpa`
- Voice 이름 해석: 직접 매칭 → OpenAI 매핑 → 대소문자 무시 매칭 → 기본 스피커
- ref_audio 처리: base64 디코딩 → 임시 파일 저장 (tempfile context manager) → 사용 후 자동 삭제
- ref_audio 프리셋: `REF_AUDIO_DIR` 내 파일명과 매칭하여 사전 등록 음성 사용 가능

### GPU 메모리 관리 전략

```
모델 사이즈별 예상 VRAM (bfloat16):
- 1.7B 모델 1개: ~3.5GB
- 0.6B 모델 1개: ~1.5GB
- Speech Tokenizer (12Hz): ~1GB (모델 간 공유 불가, 각 모델에 내장)

조합별 예상 VRAM:
- MODEL_TYPE=1.7b + TTS_MODES=all → 3개 × ~4.5GB = ~13.5GB
- MODEL_TYPE=0.6b + TTS_MODES=all → 2개 × ~2.5GB = ~5GB (VoiceDesign 없음)
- MODEL_TYPE=both + TTS_MODES=all → 5개 = ~18.5GB

권장 설정:
- 24GB GPU (RTX 4090 등): MODEL_TYPE=1.7b + TTS_MODES=all
- 48GB+ GPU (A6000, DGX 등): MODEL_TYPE=both + TTS_MODES=all
- 16GB GPU: MODEL_TYPE=0.6b 또는 MODEL_TYPE=1.7b + TTS_MODES=custom_voice
```

### VoiceClone 프리셋 관리

`REF_AUDIO_DIR`에 미리 저장된 참조 오디오를 프리셋으로 사용 가능:

```
/models/ref_audio/
├── john.wav              # "john" 프리셋
├── john.txt              # john.wav의 텍스트 (ICL 모드용, 선택)
├── sarah.wav             # "sarah" 프리셋
├── sarah.txt
└── custom_speaker.pt     # create_voice_clone_prompt()로 저장된 프리셋
```

- `ref_audio` 필드에 base64 대신 프리셋 이름(예: `"john"`)을 전달하면 해당 파일 사용
- `.pt` 파일: `Qwen3TTSModel.create_voice_clone_prompt()`로 사전 생성된 VoiceClonePromptItem → 재사용 시 추론 속도 향상
- 프리셋 이름 해석 순서: `.pt` 파일 → `.wav` 파일 → base64 디코딩

---

## 3단계: API 라우트 (`app/routes/tts.py`)

### 엔드포인트

#### `POST /v1/audio/speech`
- TTSRequest JSON body 수신
- model 이름에서 모드/사이즈 파싱 → 적절한 TTSService 호출
- VoiceDesign 모드: `instruct` 필드 필수 검증 (없으면 HTTP 400)
- VoiceClone 모드: `ref_audio` 필드 필수 검증 (없으면 HTTP 400)
- VoiceClone ICL 모드: `x_vector_only=False`일 때 `ref_text` 필수 검증
- 0.6B + VoiceDesign 요청: HTTP 400 반환 ("0.6B 모델은 VoiceDesign을 지원하지 않습니다")
- 오디오 바이트 응답 (Content-Type: audio/wav 등)

#### `GET /v1/audio/voices`
- 로드된 모델들의 사용 가능한 스피커 목록 반환
- CustomVoice: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- VoiceClone 프리셋: `REF_AUDIO_DIR`에서 스캔된 프리셋 목록 포함

#### `GET /v1/audio/languages`
- 지원 언어 목록 반환
- Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

---

## 4단계: FastAPI 앱 (`app/main.py`)

VibeVoice 패턴 그대로 적용:

```python
import logging

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Qwen3-TTS API Server...")
    init_tts_services()   # 시작 시 모델 로드
    logger.info("All TTS services initialized.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Qwen3-TTS OpenAI API Server", lifespan=lifespan)
```

**포함 항목:**
- CORS 미들웨어 (allow all)
- `X-Process-Time` 헤더 미들웨어
- OpenAI 스타일 글로벌 예외 핸들러 (`{"error": {"message": ..., "type": ..., "code": ...}}`)
- `GET /` — API 정보
- `GET /health` — 서비스 상태 (모드별, 사이즈별)
- `GET /v1/models` — 로드된 모델 목록 (OpenAI 형식)
- TTS 라우터 마운트
- 구조화된 로깅 (startup, 요청 처리, 에러)

---

## 5단계: Docker 설정

### `requirements.txt`

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6
scipy>=1.11.0
numpy>=1.24.0
```

> `qwen-tts`, `torch`, `transformers` 등은 Dockerfile에서 직접 설치 (NVIDIA 베이스 이미지에 torch 포함)

### `Dockerfile`

Qwen3-TTS 소스는 `pip install qwen-tts`로 PyPI에서 설치합니다.
(로컬 소스 COPY 방식은 빌드 컨텍스트 문제가 있으므로 PyPI 우선, 필요시 로컬 설치 옵션도 제공)

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 시스템 의존성
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# qwen-tts 패키지 설치 (PyPI에서)
RUN pip install --no-cache-dir qwen-tts

# Flash Attention 설치
RUN MAX_JOBS=4 pip install --no-cache-dir -U flash-attn --no-build-isolation

# 앱 의존성
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 앱 코드 복사
COPY app/ /workspace/app/

WORKDIR /workspace

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

> **로컬 소스 설치 옵션 (대안):**
> 빌드 컨텍스트에 Qwen3-TTS 소스를 포함시킬 경우:
> ```dockerfile
> COPY Qwen3-TTS/ /opt/Qwen3-TTS/
> RUN pip install --no-cache-dir -e /opt/Qwen3-TTS/
> ```
> 이 경우 빌드 전 `cp -r /home/onion/Workspace/Qwen3-TTS ./Qwen3-TTS` 또는
> `.dockerignore`에서 제외하지 않도록 주의 필요.

### `docker-compose.yml`

```yaml
services:
  qwen3-tts-api:
    build:
      context: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MODEL_TYPE=${MODEL_TYPE:-1.7b}
      - TTS_MODES=${TTS_MODES:-all}
      - DEVICE=${DEVICE:-cuda}
      - DTYPE=${DTYPE:-bfloat16}
      - ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
      - CUSTOM_VOICE_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
      - VOICE_DESIGN_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
      - VOICE_CLONE_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base
      - CUSTOM_VOICE_0_6B_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
      - VOICE_CLONE_0_6B_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-Base
      - DEFAULT_SPEAKER=${DEFAULT_SPEAKER:-Vivian}
      - DEFAULT_LANGUAGE=${DEFAULT_LANGUAGE:-Auto}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    volumes:
      - ${MODEL_DIR}:/models:ro
      - ${REF_AUDIO_DIR:-./ref_audio}:/models/ref_audio:ro
    ports:
      - "${API_PORT:-8080}:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### `docker-compose.with-openwebui.yml`

```yaml
services:
  qwen3-tts-api:
    build:
      context: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MODEL_TYPE=${MODEL_TYPE:-1.7b}
      - TTS_MODES=${TTS_MODES:-custom_voice}
      - CUSTOM_VOICE_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
    volumes:
      - ${MODEL_DIR}:/models:ro
    ports:
      - "${API_PORT:-8080}:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      start_period: 180s
      retries: 3

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - AUDIO_TTS_ENGINE=openai
      - AUDIO_TTS_OPENAI_API_BASE_URL=http://qwen3-tts-api:8080/v1
      - AUDIO_TTS_OPENAI_API_KEY=sk-unused
      - AUDIO_TTS_MODEL=qwen3-tts-1.7b
      - AUDIO_TTS_VOICE=Vivian
    volumes:
      - open-webui-data:/app/backend/data
    depends_on:
      qwen3-tts-api:
        condition: service_healthy

volumes:
  open-webui-data:
```

### `.env.example`

```env
# === 모델 경로 (필수) ===
MODEL_DIR=/path/to/downloaded/models
REF_AUDIO_DIR=./ref_audio

# === 서버 설정 ===
API_PORT=8080
LOG_LEVEL=INFO

# === 모델 설정 ===
MODEL_TYPE=1.7b            # 0.6b | 1.7b | both
TTS_MODES=all              # custom_voice | voice_design | voice_clone | all
DEVICE=cuda
DTYPE=bfloat16             # bfloat16 | float16
ATTN_IMPLEMENTATION=flash_attention_2   # flash_attention_2 | sdpa

# === 기본값 ===
DEFAULT_SPEAKER=Vivian
DEFAULT_LANGUAGE=Auto

# === 생성 파라미터 ===
TEMPERATURE=0.9
TOP_K=50
TOP_P=1.0
REPETITION_PENALTY=1.05
MAX_NEW_TOKENS=2048
SUBTALKER_DOSAMPLE=true
SUBTALKER_TOP_K=50
SUBTALKER_TOP_P=1.0
SUBTALKER_TEMPERATURE=0.9
```

---

## 6단계: 자동화 테스트 스크립트 (`run-test.sh`)

VibeVoice의 `run-test.sh` 패턴을 따라 작성:

```bash
#!/bin/bash
# 기능:
# 1. CLI 옵션 파싱 (--model, --modes, --port, --skip-build, --down)
# 2. Docker/NVIDIA 사전 검증
# 3. 모델 디렉토리 존재 확인
# 4. docker compose build && up
# 5. health 엔드포인트 대기 (최대 300초)
# 6. 자동화 테스트:
#    - GET /health
#    - GET /v1/models
#    - GET /v1/audio/voices
#    - GET /v1/audio/languages
#    - POST /v1/audio/speech (CustomVoice, 한국어)
#    - POST /v1/audio/speech (CustomVoice, OpenAI 호환 tts-1-hd + alloy)
#    - POST /v1/audio/speech (VoiceDesign, 활성화 시)
#    - POST /v1/audio/speech (VoiceClone, 활성화 시)
#    - POST /v1/audio/speech (빈 input → 400 검증)
#    - POST /v1/audio/speech (존재하지 않는 모델 → 400 검증)
# 7. 결과 리포트
```

---

## 모델 다운로드 가이드

로컬 모델 전용이므로, 사전에 HuggingFace에서 다운로드 필요:

```bash
# 1.7B 모델 (CustomVoice + VoiceDesign + Base)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --local-dir /path/to/models/Qwen3-TTS-12Hz-1.7B-CustomVoice

huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --local-dir /path/to/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign

huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local-dir /path/to/models/Qwen3-TTS-12Hz-1.7B-Base

# 0.6B 모델 (CustomVoice + Base만, VoiceDesign 없음)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --local-dir /path/to/models/Qwen3-TTS-12Hz-0.6B-CustomVoice

huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir /path/to/models/Qwen3-TTS-12Hz-0.6B-Base
```

> **디렉토리 구조 예시:**
> ```
> /path/to/models/
> ├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
> ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
> ├── Qwen3-TTS-12Hz-1.7B-Base/
> ├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
> └── Qwen3-TTS-12Hz-0.6B-Base/
> ```
> `.env`의 `MODEL_DIR`에 이 상위 경로를 지정합니다.

---

## 구현 순서

| 순서 | 파일 | 설명 |
|------|------|------|
| 1 | `app/__init__.py` | 패키지 초기화, 버전 |
| 2 | `app/config.py` | 환경변수 설정 클래스 |
| 3 | `app/models/__init__.py`, `app/models/tts_models.py` | Pydantic 스키마 |
| 4 | `app/services/__init__.py`, `app/services/tts_service.py` | TTS 추론 서비스 |
| 5 | `app/routes/__init__.py`, `app/routes/tts.py` | API 엔드포인트 |
| 6 | `app/main.py` | FastAPI 앱 조립 |
| 7 | `requirements.txt` | Python 의존성 |
| 8 | `.env.example` | 환경변수 템플릿 |
| 9 | `.gitignore` | Git 무시 파일 |
| 10 | `Dockerfile` | Docker 이미지 빌드 |
| 11 | `docker-compose.yml` | Docker Compose 배포 |
| 12 | `docker-compose.with-openwebui.yml` | Open WebUI 연동 배포 |
| 13 | `run-test.sh` | 자동화 테스트 스크립트 |

---

## 검증 방법

### 로컬 테스트 (Docker 없이)
```bash
cd /mnt/USERS/onion/DATA_ORIGN/Workspace/Qwen3TTSDockerOpenApiServer
pip install -e /home/onion/Workspace/Qwen3-TTS
pip install -r requirements.txt

# 환경변수 설정
export CUSTOM_VOICE_MODEL_PATH=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
export MODEL_TYPE=1.7b
export TTS_MODES=custom_voice

python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### API 테스트
```bash
# Health check
curl http://localhost:8080/health

# 모델 목록
curl http://localhost:8080/v1/models

# 음성 목록
curl http://localhost:8080/v1/audio/voices

# 지원 언어 목록
curl http://localhost:8080/v1/audio/languages

# CustomVoice TTS
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b","input":"안녕하세요","voice":"Vivian"}' \
  --output test_custom.wav

# CustomVoice + instruct (스타일 지시)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b","input":"안녕하세요","voice":"Vivian","instruct":"매우 밝고 행복한 톤으로"}' \
  --output test_custom_instruct.wav

# OpenAI 호환 요청
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1-hd","input":"Hello world","voice":"alloy"}' \
  --output test_openai.wav

# VoiceDesign TTS
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b-design","input":"Hello!","instruct":"A warm friendly female voice"}' \
  --output test_design.wav

# VoiceClone TTS - ICL 모드 (ref_text 필수)
REF_B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-tts-1.7b-clone\",\"input\":\"Hello\",\"ref_audio\":\"${REF_B64}\",\"ref_text\":\"Reference text\"}" \
  --output test_clone_icl.wav

# VoiceClone TTS - X-vector only 모드 (ref_text 불필요)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-tts-1.7b-clone\",\"input\":\"Hello\",\"ref_audio\":\"${REF_B64}\",\"x_vector_only\":true}" \
  --output test_clone_xvec.wav

# VoiceClone TTS - 프리셋 사용
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b-clone","input":"Hello","ref_audio":"john"}' \
  --output test_clone_preset.wav

# 에러 케이스: 빈 입력
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b","input":""}' \
  -w "\nHTTP Status: %{http_code}\n"
```

### Docker 테스트
```bash
docker compose up --build
# 동일한 curl 명령으로 테스트
```

### 자동화 테스트
```bash
./run-test.sh --model 1.7b --modes all --port 8080
```

---

## 주요 참조 파일

| 목적 | 파일 경로 |
|------|-----------|
| Qwen3-TTS 추론 API | `/home/onion/Workspace/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py` |
| Qwen3-TTS 패키지 설정 | `/home/onion/Workspace/Qwen3-TTS/pyproject.toml` |
| Qwen3-TTS 예제 (CustomVoice) | `/home/onion/Workspace/Qwen3-TTS/examples/test_model_12hz_custom_voice.py` |
| Qwen3-TTS 예제 (VoiceClone) | `/home/onion/Workspace/Qwen3-TTS/examples/test_model_12hz_base.py` |
| Qwen3-TTS 예제 (VoiceDesign) | `/home/onion/Workspace/Qwen3-TTS/examples/test_model_12hz_voice_design.py` |
| Qwen3-TTS Gradio 데모 | `/home/onion/Workspace/Qwen3-TTS/qwen_tts/cli/demo.py` |
| VibeVoice 참조 서버 (main) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/app/main.py` |
| VibeVoice 참조 서버 (tts route) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/app/routes/tts.py` |
| VibeVoice 참조 서버 (tts service) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/app/services/tts_service.py` |
| VibeVoice 참조 서버 (config) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/app/config.py` |
| VibeVoice 참조 서버 (models) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/app/models/tts_models.py` |
| VibeVoice 참조 서버 (Dockerfile) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/Dockerfile` |
| VibeVoice 참조 서버 (run-test.sh) | `/home/onion/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/run-test.sh` |
