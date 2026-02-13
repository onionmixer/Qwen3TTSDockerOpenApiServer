# Qwen3-TTS Docker OpenAI API Server

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba)를 OpenAI TTS API 규격으로 서빙하는 Docker 프로젝트입니다.

OpenAI의 `/v1/audio/speech` API를 사용하는 클라이언트(Open WebUI, ChatBox 등)에서 Qwen3-TTS를 **드롭인 대체(drop-in replacement)**로 사용할 수 있습니다.

## 원본 프로젝트

- **Qwen3-TTS** : https://github.com/QwenLM/Qwen3-TTS
- Alibaba Qwen 팀이 개발한 오픈소스 Text-to-Speech 모델
- 12Hz 코덱 기반, CustomVoice / VoiceDesign / VoiceClone 3가지 모드 지원
- 0.6B (경량) 및 1.7B (고품질) 두 가지 모델 사이즈 제공

본 프로젝트는 Qwen3-TTS의 추론 API를 래핑하여 OpenAI 호환 REST API로 노출하는 **서빙 레이어**이며, **커스텀 음성 파인튜닝** 파이프라인도 포함합니다.

## 지원 기능

| 기능 | 설명 |
|------|------|
| **CustomVoice** | 내장 스피커(Vivian, Ryan, Serena 등 9종)로 TTS |
| **VoiceDesign** | 자연어 지시문으로 새로운 음성 스타일 생성 (1.7B만) |
| **VoiceClone** | 참조 오디오로 음성 복제 |
| **듀얼 모델** | 0.6B / 1.7B 동시 로드 가능 |
| **OpenAI 호환** | `tts-1`, `tts-1-hd` 모델명 및 `alloy`~`shimmer` 음성 매핑 |
| **다중 포맷** | WAV, MP3, Opus, AAC, FLAC, PCM 출력 |
| **음성 파인튜닝** | Base 모델로 커스텀 화자 학습 (0.6B/1.7B) |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/v1/audio/speech` | 텍스트를 음성으로 변환 (OpenAI 호환) |
| `GET` | `/v1/audio/voices` | 사용 가능한 음성 목록 |
| `GET` | `/v1/models` | 로드된 모델 목록 (OpenAI 호환) |
| `GET` | `/health` | 서비스 상태 확인 |
| `GET` | `/docs` | Swagger UI |

## TTS 모드 비교

Qwen3-TTS는 3가지 TTS 모드를 제공하며, 각 모드는 별도의 모델 파일을 사용합니다.

| | **CustomVoice** | **VoiceDesign** | **VoiceClone** (Base) |
|---|---|---|---|
| **용도** | 내장 화자로 TTS | 자연어로 새 음성 생성 | 참조 오디오로 음성 복제 |
| **입력** | 텍스트 + 화자 이름 | 텍스트 + 음성 설명문 | 텍스트 + 참조 오디오(WAV) |
| **화자 선택** | 9종 내장 (Vivian, Ryan 등) | 없음 (설명문으로 생성) | 없음 (참조 오디오로 결정) |
| **1.7B 모델** | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `Qwen3-TTS-12Hz-1.7B-Base` |
| **0.6B 모델** | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 없음 (1.7B만) | `Qwen3-TTS-12Hz-0.6B-Base` |
| **VRAM (1.7B)** | ~4.4GB | ~4.4GB | ~4.4GB |
| **파인튜닝** | X (파인튜닝 결과물) | X | O (파인튜닝 원본) |
| **OpenAI 호환** | O (`tts-1`, `tts-1-hd`, `alloy` 등) | X (확장 API) | X (확장 API) |

> **VRAM 참고:** 각 모드는 별도의 모델을 GPU에 로드합니다. 1.7B 모델 1개당 ~4.4GB VRAM을 사용하므로, 12GB GPU에서는 최대 2개 모드까지 동시 사용 가능합니다.

**모드별 사용 예시:**

- **CustomVoice**: 가장 일반적인 TTS. OpenAI 호환 클라이언트(Open WebUI 등)에서 사용. `instruct`로 스타일 지시 가능 ("밝고 행복한 톤으로")
- **VoiceDesign**: 자연어 설명으로 원하는 음성 특성 지정 ("A deep, warm male voice"). 매번 다른 음성이 생성될 수 있음
- **VoiceClone**: 참조 오디오의 음성을 복제하여 새 텍스트를 읽음. 파인튜닝의 시작점이 되어 CustomVoice 모델로 변환 가능

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

**전체 모델 일괄 다운로드** (~18.6GB):

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
| 파인튜닝 (공통) | `Qwen3-TTS-Tokenizer-12Hz` | ~651MB |

> 추론용 Tokenizer는 각 모델 내부에 포함되어 있어 별도 다운로드 불필요합니다. 파인튜닝 시에만 `Qwen3-TTS-Tokenizer-12Hz`가 필요합니다.

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
│   ├── model.safetensors
│   ├── speech_tokenizer/
│   └── ...
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/  (선택)
├── Qwen3-TTS-12Hz-1.7B-Base/         (선택)
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/  (선택)
├── Qwen3-TTS-12Hz-0.6B-Base/         (선택)
└── Qwen3-TTS-Tokenizer-12Hz/         (파인튜닝 시 필요)
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
TTS_MODES=custom_voice,voice_clone

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
[TTS-voice_clone-1.7b] Model loaded successfully
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

> **주의:** VoiceDesign을 사용하려면 `TTS_MODES`에 `voice_design`이 포함되어야 합니다. 기본 설정(`custom_voice,voice_clone`)에는 포함되어 있지 않으므로, `.env`에서 `TTS_MODES=custom_voice,voice_clone,voice_design` 또는 `TTS_MODES=all`로 변경하세요. 3개 모드 동시 로드에는 약 14GB VRAM이 필요합니다.

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
| `TTS_MODES` | `custom_voice,voice_clone` | 활성화할 모드 (`custom_voice`, `voice_design`, `voice_clone`, `all`) |
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
| 1.7B CustomVoice 단독 | ~4.5GB | 8GB+ |
| 1.7B CustomVoice + VoiceClone (기본값) | **~9.5GB** | **12GB+** |
| 1.7B CustomVoice + VoiceDesign | ~9.0GB | 12GB+ |
| 0.6B CustomVoice 단독 | ~2.5GB | 6GB+ |
| 1.7B 전체 모드 (custom+design+clone) | ~13.5GB | 24GB+ |
| 0.6B + 1.7B 전체 | ~18.5GB | 24GB+ |

> **검증 환경:** RTX 3080 Ti (12GB) + 1.7B CustomVoice + VoiceClone = VRAM 9,702 MiB 사용 (여유 ~2.5GB)

### Pascal GPU (GTX 1080 Ti 등) - CPU 모드

Pascal 세대 GPU는 CUDA 모드 불가. CPU 모드로 실행:

```env
DEVICE=cpu
DTYPE=float32
ATTN_IMPLEMENTATION=sdpa
MODEL_TYPE=0.6b
```

> CPU 모드에서는 0.6B 모델 기준 문장당 약 10~15초 소요. 1.7B는 0.6B 사용 권장.

## 음성 파인튜닝 (Custom Voice Fine-tuning)

Qwen3-TTS Base 모델을 사용하여 특정 화자의 음성으로 파인튜닝할 수 있습니다. 파인튜닝된 모델은 CustomVoice 모드로 변환되어, 참조 오디오 없이도 학습된 화자의 음성으로 TTS를 수행합니다.

### 파인튜닝 개요

| 항목 | 설명 |
|------|------|
| **지원 모델** | 1.7B-Base (권장), 0.6B-Base |
| **학습 방식** | 단일 화자 SFT (Supervised Fine-Tuning) |
| **출력** | CustomVoice 타입 모델 (참조 오디오 불필요) |
| **필요 데이터** | WAV 오디오 + 텍스트 전사 (JSONL) |
| **VRAM 요구량** | 0.6B: ~6GB (12GB GPU OK), 1.7B: **24GB+ 필요** |

> **RTX 3080 Ti (12GB) 검증 결과:** 0.6B-Base 모델 파인튜닝 성공 (batch_size=2). 1.7B-Base는 모델만으로 ~9.8GB를 사용하여 optimizer 상태를 위한 VRAM이 부족합니다.

### VRAM별 권장 설정

| GPU VRAM | 모델 | batch_size | 비고 |
|----------|------|------------|------|
| 12GB (RTX 3080 Ti 등) | 0.6B-Base | 2 | 기본 AdamW 사용 가능 |
| 16GB (RTX 4080 등) | 0.6B-Base | 4~8 | 여유 있음 |
| 24GB+ (RTX 3090, A100 등) | 1.7B-Base | 2~8 | 권장, 고품질 |

### 파인튜닝 파이프라인

#### 1단계: 학습 데이터 준비

JSONL 형식으로 학습 데이터를 준비합니다. 각 행에는 오디오 파일 경로, 텍스트 전사, 참조 오디오 경로가 필요합니다.

```jsonl
{"audio":"./data/utt0001.wav","text":"안녕하세요, 오늘 날씨가 좋습니다.","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

**데이터 요구사항:**
- 오디오: 24kHz WAV 형식 (mono)
- `ref_audio`: 모든 샘플에 동일한 참조 오디오 사용 권장 (화자 일관성 향상)
- 최소 50~100개 이상의 utterance 권장 (적은 데이터로는 품질 저하)
- 텍스트는 해당 오디오의 정확한 전사여야 함

#### 2단계: 오디오 코드 추출 (prepare_data)

Qwen3-TTS Tokenizer를 사용하여 오디오를 이산 코드(discrete codes)로 변환합니다.

```bash
cd Qwen3-TTS/finetuning

python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ../../models/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

> Tokenizer를 별도 다운로드하지 않았다면, Base 모델 경로를 사용해도 됩니다 (내부 `speech_tokenizer` 포함).

**별도 다운로드가 필요한 경우:**
```bash
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --local-dir ./models/Qwen3-TTS-Tokenizer-12Hz
```

#### 3단계: 파인튜닝 실행

> **0.6B 모델 사용 시:** 원본 `Qwen3-TTS/finetuning/sft_12hz.py`는 1.7B 전용입니다. 0.6B 모델에서는 차원 불일치 에러가 발생하므로, 본 프로젝트의 수정된 스크립트(`finetuning_test/sft_12hz.py`)를 사용하세요. 자세한 내용은 아래 [0.6B 모델 학습 시 주의사항](#06b-모델-학습-시-주의사항)을 참조하세요.

```bash
# 0.6B: 수정된 스크립트 사용 (프로젝트 루트에서)
cd finetuning_test

python sft_12hz.py \
  --init_model_path ../models/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path ./output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name my_speaker
```

```bash
# 1.7B: 원본 스크립트 사용 (24GB+ VRAM 필요)
cd Qwen3-TTS/finetuning

python sft_12hz.py \
  --init_model_path ../../models/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name my_speaker
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--init_model_path` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 초기 모델 경로 |
| `--batch_size` | 2 | 배치 크기 (VRAM에 맞게 조절) |
| `--lr` | 2e-5 | 학습률 |
| `--num_epochs` | 3 | 학습 에폭 수 |
| `--speaker_name` | `speaker_test` | 커스텀 스피커 이름 |

**학습 진행 시 출력 예시:**
```
Epoch 0 | Step 0 | Loss: 13.8789
Epoch 1 | Step 0 | Loss: 8.5651
Epoch 2 | Step 0 | Loss: 6.3911
```

체크포인트는 `output/checkpoint-epoch-{N}/` 에 저장됩니다.

#### 4단계: 추론 테스트

파인튜닝된 체크포인트에서 음성을 생성합니다.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # flash_attention_2 if available
)

wavs, sr = tts.generate_custom_voice(
    text="안녕하세요, 파인튜닝된 음성입니다.",
    speaker="my_speaker",
)
sf.write("output.wav", wavs[0], sr)
```

### 파인튜닝 팁

- **데이터 양**: 최소 50~100 utterance 이상 권장. 8개 샘플로는 일부 언어에서 품질 저하 발생 확인
- **참조 오디오**: 모든 학습 샘플에 동일한 ref_audio 사용 시 화자 일관성 향상
- **에폭 수**: 데이터가 적으면 3~5 에폭, 데이터가 많으면 10+ 에폭 권장
- **학습률**: 대규모 데이터셋(1000+)은 2e-6, 소규모(~100)는 2e-5 권장
- **0.6B vs 1.7B**: 0.6B는 12GB GPU에서 학습 가능하나 품질은 1.7B 대비 낮음. 충분한 VRAM(24GB+)이 있으면 1.7B 권장
- **Flash Attention 미설치 시**: `sdpa`로 자동 fallback. Docker 환경에서는 Flash Attention 2 포함

### 파인튜닝 모델을 API 서버에서 사용

파인튜닝된 체크포인트를 `models/` 디렉토리에 복사하고, `docker-compose.yml`의 환경변수를 수정하여 사용할 수 있습니다:

```yaml
environment:
  - CUSTOM_VOICE_MODEL_PATH=/models/my-finetuned-model
```

파인튜닝된 모델은 CustomVoice 타입으로 변환되므로, `qwen3-tts-1.7b` 또는 `qwen3-tts-0.6b` 모델명으로 요청하면 됩니다.

```bash
curl -X POST http://localhost:8899/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-0.6b","input":"안녕하세요","voice":"my_speaker"}' \
  --output output.wav
```

### 0.6B 모델 학습 시 주의사항

공식 `sft_12hz.py` 스크립트는 1.7B 모델 기준으로 작성되어 있어, 0.6B 모델에서는 `text_hidden_size`(2048)와 `hidden_size`(1024) 차이로 인한 차원 불일치 에러가 발생합니다. 이를 해결하려면 학습 스크립트에서 text_embedding에 `text_projection`을 적용해야 합니다:

```python
# 원본 (1.7B 전용)
input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask

# 수정 (0.6B 호환)
raw_text_embedding = model.talker.model.text_embedding(input_text_ids)
input_text_embedding = model.talker.text_projection(raw_text_embedding) * text_embedding_mask
```

본 프로젝트의 `finetuning_test/sft_12hz.py`에는 이 수정이 적용되어 있습니다.

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
├── finetuning_test/             # 파인튜닝 테스트 (0.6B 검증 완료)
│   ├── sft_12hz.py              # 수정된 학습 스크립트 (0.6B 호환)
│   ├── dataset.py               # 학습 데이터셋 클래스
│   ├── generate_samples.py      # 샘플 데이터 생성 도구
│   ├── test_inference.py        # 파인튜닝 모델 추론 테스트
│   └── data/                    # 샘플 오디오 데이터
├── Qwen3-TTS/                   # Qwen3-TTS 소스 (git clone)
├── models/                      # 다운로드한 모델 (git 미추적)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── start.sh                    # Docker 컨테이너 시작 스크립트
└── stop.sh                     # Docker 컨테이너 종료 스크립트
```

## 라이선스

이 서빙 레이어는 원본 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)의 Apache 2.0 라이선스를 따릅니다.
