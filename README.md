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

## OpenAI 음성 매핑

| OpenAI voice | Qwen3 speaker | 설명 |
|-------------|---------------|------|
| `alloy` | `Vivian` | 밝고 또렷한 여성 |
| `echo` | `Ryan` | 역동적인 남성 |
| `fable` | `Serena` | 따뜻하고 부드러운 여성 |
| `onyx` | `Uncle_Fu` | 깊고 낮은 남성 |
| `nova` | `Ono_Anna` | 발랄한 일본어 여성 |
| `shimmer` | `Sohee` | 감성적인 한국어 여성 |

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

> Tokenizer는 각 모델 내부에 포함되어 있어 별도 다운로드 불필요합니다.

**개별 다운로드 예시** (CustomVoice 1.7B만):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

다운로드 후 디렉토리 구조:

```
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── ...
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/  (선택)
├── Qwen3-TTS-12Hz-1.7B-Base/         (선택)
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/  (선택)
└── Qwen3-TTS-12Hz-0.6B-Base/         (선택)
```

### 2. Qwen3-TTS 소스 준비

Dockerfile에서 로컬 소스를 빌드하므로 프로젝트 루트에 `Qwen3-TTS/` 디렉토리가 필요합니다:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
```

### 3. Docker Compose 실행

#### 3-1. 환경 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집합니다:

```env
# 모델 디렉토리 절대 경로 (필수)
MODEL_DIR=/home/user/project/models

# 모델 사이즈: 0.6b, 1.7b, both
MODEL_TYPE=1.7b

# 활성화할 모드: custom_voice, voice_design, voice_clone, all
TTS_MODES=custom_voice

# 서버 포트 (기본: 8899)
API_PORT=8899

# GPU 설정 (bfloat16 권장)
DEVICE=cuda
DTYPE=bfloat16
ATTN_IMPLEMENTATION=flash_attention_2
```

> `MODEL_DIR` 아래에 다운로드한 모델 디렉토리들이 있어야 합니다. Docker Compose에서 이 경로를 컨테이너 내부 `/models`로 읽기 전용 마운트합니다.

#### 3-2. 빌드 및 실행

```bash
docker compose up --build -d
```

#### 3-3. 로그 확인

```bash
docker compose logs -f
```

모델 로딩이 완료되면 다음과 같은 메시지가 표시됩니다:

```
[TTS-custom_voice-1.7b] Model loaded successfully
[TTS-custom_voice-1.7b] Speakers: ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']
[Startup] TTS services ready
Server ready at http://0.0.0.0:8080
```

#### 3-4. VoiceClone 참조 오디오 (선택)

VoiceClone 모드에서 파일 기반 참조 오디오를 사용하려면:

```env
REF_AUDIO_DIR=/home/user/ref_audio
```

이 디렉토리는 컨테이너 내부 `/ref_audio`로 마운트됩니다.

### 4. 로컬 실행 (Docker 없이)

```bash
# Qwen3-TTS 패키지 설치
pip install -e ./Qwen3-TTS

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
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b","input":"안녕하세요, 반갑습니다.","voice":"Vivian"}' \
  --output output.wav
```

### OpenAI 호환 요청

```bash
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1-hd","input":"Hello world","voice":"alloy"}' \
  --output output.wav
```

### CustomVoice + 스타일 지시 (instruct)

```bash
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b","input":"안녕하세요","voice":"Vivian","instruct":"매우 밝고 행복한 톤으로"}' \
  --output output.wav
```

> 참고: 0.6B 모델에서는 `instruct`가 무시됩니다.

### VoiceDesign (1.7B 전용)

```bash
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b-design","input":"Hello!","instruct":"A warm friendly female voice"}' \
  --output output.wav
```

### VoiceClone

```bash
# base64 인코딩된 참조 오디오 사용
REF_B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-tts-1.7b-clone\",\"input\":\"Hello\",\"ref_audio\":\"${REF_B64}\",\"ref_text\":\"Reference transcript\"}" \
  --output output.wav
```

### Python 클라이언트

```python
import requests

response = requests.post("http://localhost:8899/v1/audio/speech",
    json={
        "model": "tts-1-hd",
        "input": "안녕하세요, 반갑습니다.",
        "voice": "alloy",
        "response_format": "wav",
    })

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8899/v1", api_key="unused")

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Hello world, nice to meet you!",
)
response.stream_to_file("output.wav")
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_DIR` | - | 모델 디렉토리 경로 (필수) |
| `MODEL_TYPE` | `1.7b` | 로드할 모델 사이즈 (`0.6b`, `1.7b`, `both`) |
| `TTS_MODES` | `custom_voice` | 활성화할 모드 (`custom_voice`, `voice_design`, `voice_clone`, `all`) |
| `DEVICE` | `cuda` | PyTorch 디바이스 (`cuda`, `cpu`) |
| `DTYPE` | `bfloat16` | 모델 정밀도 (`bfloat16`, `float16`) |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention 구현 (`flash_attention_2`, `sdpa`) |
| `DEFAULT_SPEAKER` | `Vivian` | 기본 스피커 |
| `DEFAULT_LANGUAGE` | `Auto` | 기본 언어 |
| `API_PORT` | `8899` | 서버 포트 |
| `REF_AUDIO_DIR` | `./ref_audio` | VoiceClone 참조 오디오 디렉토리 |

전체 목록은 [.env.example](.env.example) 참조.

## GPU 호환성

### dtype 설정

| dtype | 설명 | Flash Attention 2 | 권장 |
|-------|------|-------------------|------|
| `bfloat16` | **권장**. Ampere(30xx) 이상에서 네이티브 지원 | O | **RTX 3080 Ti에서 검증 완료** |
| `float16` | Volta(V100)/Turing(20xx)부터 지원 | O | bfloat16 미지원 GPU용 |
| `float32` | Flash Attention 2와 **호환 불가** | X | 사용 불가 |

> **중요:** Flash Attention 2는 `float16`과 `bfloat16`만 지원합니다. `float32`로 설정 시 모델 로드는 되지만 추론 시 에러가 발생합니다.

### 지원 GPU

| GPU 세대 | Compute Capability | bfloat16 | 지원 여부 |
|----------|-------------------|----------|-----------|
| Pascal (GTX 1080 Ti 등) | sm_61 | X | **미지원** (CUDA) |
| Volta (V100) | sm_70 | X | 지원 (float16 사용) |
| Turing (RTX 20xx) | sm_75 | X | 지원 (float16 사용) |
| Ampere (RTX 30xx, A100) | sm_80/86 | **O** | **지원 (bfloat16 권장)** |
| Ada Lovelace (RTX 40xx) | sm_89 | **O** | 지원 (bfloat16 권장) |
| Hopper (H100) | sm_90 | **O** | 지원 (bfloat16 권장) |

### VRAM 요구사항

| 구성 | 예상 VRAM | 권장 GPU |
|------|-----------|----------|
| 1.7B CustomVoice | ~4.5GB | 8GB+ |
| 0.6B CustomVoice | ~2.5GB | 6GB+ |
| 1.7B 전체 모드 (custom+design+clone) | ~13.5GB | 24GB+ |
| 0.6B + 1.7B 전체 | ~18.5GB | 24GB+ |

> **검증 환경:** RTX 3080 Ti (12GB) + 1.7B CustomVoice bfloat16 = VRAM 4,440 MiB 사용 (여유 ~6.8GB)

### Pascal GPU (GTX 1080 Ti 등) - CPU 모드

Pascal 세대 GPU는 CUDA 모드 불가. CPU 모드로 실행:

```env
DEVICE=cpu
DTYPE=float32
ATTN_IMPLEMENTATION=sdpa
MODEL_TYPE=0.6b
```

> CPU 모드에서는 0.6B 모델 기준 문장당 약 10~15초 소요. 1.7B는 0.6B 사용 권장.

## 프로젝트 구조

```
Qwen3TTSDockerOpenApiServer/
├── app/
│   ├── __init__.py              # 버전 정보
│   ├── config.py                # pydantic-settings 환경변수 설정
│   ├── main.py                  # FastAPI 앱, lifespan, 미들웨어
│   ├── routes/
│   │   └── tts.py               # POST /v1/audio/speech, GET /v1/audio/voices
│   ├── services/
│   │   └── tts_service.py       # Qwen3TTSModel 래핑, 추론, 오디오 변환
│   └── models/
│       └── tts_models.py        # TTSRequest, VoiceInfo Pydantic 스키마
├── Qwen3-TTS/                   # Qwen3-TTS 소스 (git clone)
├── models/                      # 다운로드한 모델 (git 미추적)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .dockerignore
```

## 라이선스

이 서빙 레이어는 원본 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)의 Apache 2.0 라이선스를 따릅니다.
