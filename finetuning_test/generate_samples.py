"""Generate sample TTS audio files for fine-tuning test."""
import requests
import json
import os

API_URL = "http://localhost:8899/v1/audio/speech"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample sentences (mixed Korean/English for variety)
samples = [
    ("utt0001", "안녕하세요, 저는 오늘 날씨가 정말 좋다고 생각합니다."),
    ("utt0002", "She said she would be here by noon, but I haven't seen her yet."),
    ("utt0003", "인공지능 기술은 우리의 일상생활을 크게 변화시키고 있습니다."),
    ("utt0004", "The quick brown fox jumps over the lazy dog near the river."),
    ("utt0005", "오늘 회의에서 중요한 결정을 내려야 합니다."),
    ("utt0006", "Technology continues to evolve at an incredible pace every year."),
    ("utt0007", "이 프로젝트는 매우 흥미롭고 도전적인 과제입니다."),
    ("utt0008", "Please make sure to review the document before the deadline tomorrow."),
]

# Generate reference audio (using same speaker for consistency)
print("Generating reference audio...")
ref_payload = {
    "model": "qwen3-tts-1.7b",
    "input": "Hello, this is a reference audio sample for voice cloning and fine-tuning.",
    "voice": "Vivian",
    "response_format": "wav"
}
resp = requests.post(API_URL, json=ref_payload)
if resp.status_code == 200:
    ref_path = os.path.join(OUTPUT_DIR, "ref.wav")
    with open(ref_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved: {ref_path} ({len(resp.content)} bytes)")
else:
    print(f"  ERROR: {resp.status_code} - {resp.text}")

# Generate training utterances
print("\nGenerating training utterances...")
for utt_id, text in samples:
    payload = {
        "model": "qwen3-tts-1.7b",
        "input": text,
        "voice": "Vivian",
        "response_format": "wav"
    }
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        wav_path = os.path.join(OUTPUT_DIR, f"{utt_id}.wav")
        with open(wav_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved: {wav_path} ({len(resp.content)} bytes)")
    else:
        print(f"  ERROR for {utt_id}: {resp.status_code} - {resp.text}")

# Create training JSONL
print("\nCreating train_raw.jsonl...")
jsonl_path = os.path.join(os.path.dirname(__file__), "train_raw.jsonl")
with open(jsonl_path, "w") as f:
    for utt_id, text in samples:
        entry = {
            "audio": os.path.join(OUTPUT_DIR, f"{utt_id}.wav"),
            "text": text,
            "ref_audio": os.path.join(OUTPUT_DIR, "ref.wav")
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"  Saved: {jsonl_path}")
print("\nDone!")
