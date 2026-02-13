"""Test inference from fine-tuned checkpoint."""
import torch
import soundfile as sf
import time

device = "cuda:0"
checkpoint_path = "/mnt/USERS/onion/DATA_ORIGN/Workspace/Qwen3TTSDockerOpenApiServer/finetuning_test/output_0.6b/checkpoint-epoch-2"

print(f"Loading model from: {checkpoint_path}")
start = time.time()

from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    checkpoint_path,
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
print(f"Model loaded in {time.time()-start:.1f}s")

# Get supported speakers
speakers = tts.get_supported_speakers()
print(f"Supported speakers: {speakers}")

# Test Korean
print("\nTest 1: Korean text with my_speaker...")
start = time.time()
wavs, sr = tts.generate_custom_voice(
    text="안녕하세요, 파인튜닝된 모델로 음성을 생성합니다.",
    speaker="my_speaker",
)
sf.write("/mnt/USERS/onion/DATA_ORIGN/Workspace/Qwen3TTSDockerOpenApiServer/finetuning_test/output_test_kr.wav", wavs[0], sr)
print(f"  Generated in {time.time()-start:.1f}s, duration: {len(wavs[0])/sr:.2f}s, sample_rate: {sr}")

# Test English
print("\nTest 2: English text with my_speaker...")
start = time.time()
wavs, sr = tts.generate_custom_voice(
    text="Hello, this is a fine-tuned model generating speech.",
    speaker="my_speaker",
)
sf.write("/mnt/USERS/onion/DATA_ORIGN/Workspace/Qwen3TTSDockerOpenApiServer/finetuning_test/output_test_en.wav", wavs[0], sr)
print(f"  Generated in {time.time()-start:.1f}s, duration: {len(wavs[0])/sr:.2f}s, sample_rate: {sr}")

print("\nDone! Check output_test_kr.wav and output_test_en.wav")
