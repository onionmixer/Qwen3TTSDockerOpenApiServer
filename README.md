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

```bash
pip install huggingface_hub
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

### 2. Docker Compose 실행

```bash
cp .env.example .env
# .env에서 MODEL_DIR을 모델 다운로드 경로로 설정

docker compose up --build
```

### 3. 로컬 실행 (Docker 없이)

```bash
pip install -e /path/to/Qwen3-TTS
pip install -r requirements.txt

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
