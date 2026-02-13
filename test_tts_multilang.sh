#!/bin/bash
#
# Qwen3-TTS Multi-Language TTS Test
#
# [내장 스피커 (CustomVoice)]
#   여성: vivian (기본값), serena, sohee, ono_anna
#   남성: ryan, eric, dylan, aiden, uncle_fu
#
# [OpenAI 호환 별칭]
#   alloy->Vivian, echo->Ryan, fable->Serena
#   onyx->Uncle_Fu, nova->Ono_Anna, shimmer->Sohee
#

API_URL="http://192.168.1.16:8899/v1/audio/speech"
#API_URL="http://192.168.1.20:8899/v1/audio/speech"
#API_URL="http://192.168.1.135:8899/v1/audio/speech"
MODEL="qwen3-tts-1.7b"
VOICE="vivian"
OUTPUT_DIR="./tts_output"

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.wav

declare -A TEXTS
TEXTS=(
  ["en"]="Hello, world! This is a text-to-speech test in English."
  ["ko"]="안녕하세요, 세계! 한국어 음성 합성 테스트입니다."
  ["ja"]="こんにちは、せかい！にほんごのおんせいごうせいてすとです。"
  ["zh"]="你好，世界！这是中文语音合成测试。"
)

declare -A LANG_NAMES
LANG_NAMES=(
  ["en"]="English"
  ["ko"]="Korean"
  ["ja"]="Japanese"
  ["zh"]="Chinese"
)

echo "=== Multi-Language TTS Test ==="
echo "Model: $MODEL | Voice: $VOICE"
echo ""

for lang in en ko ja zh; do
  text="${TEXTS[$lang]}"
  name="${LANG_NAMES[$lang]}"
  outfile="$OUTPUT_DIR/test_${lang}.wav"

  echo "[$name] Generating: $text"

  curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"input\":\"$text\",\"voice\":\"$VOICE\"}" \
    -o "$outfile" \
    -w "  -> HTTP %{http_code} | Size: %{size_download} bytes | Time: %{time_total}s\n"

  if [ -f "$outfile" ] && [ "$(wc -c < "$outfile")" -gt 0 ]; then
    echo "  -> Saved: $outfile"
  else
    echo "  -> FAILED"
  fi
  echo ""
done

# Mixed-language test
MIXED_TEXT="Hello! 안녕하세요! こんにちは！你好！This is a multilingual mixed test. 다국어 혼합 테스트입니다. たげんごこんごうてすとです。这是多语言混合测试。"
MIXED_OUT="$OUTPUT_DIR/test_mixed.wav"

echo "[Mixed] Generating: $MIXED_TEXT"

curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"input\":\"$MIXED_TEXT\",\"voice\":\"$VOICE\"}" \
  -o "$MIXED_OUT" \
  -w "  -> HTTP %{http_code} | Size: %{size_download} bytes | Time: %{time_total}s\n"

if [ -f "$MIXED_OUT" ] && [ "$(wc -c < "$MIXED_OUT")" -gt 0 ]; then
  echo "  -> Saved: $MIXED_OUT"
else
  echo "  -> FAILED"
fi
echo ""

echo "=== Done ==="
ls -lh "$OUTPUT_DIR"/*.wav 2>/dev/null
