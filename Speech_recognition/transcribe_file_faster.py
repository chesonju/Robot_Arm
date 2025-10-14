# transcribe_file_faster.py
import sys
import os
from faster_whisper import WhisperModel

try:
    from Speech_recognition.parse_floor import parse_floor
except ImportError:
    from parse_floor import parse_floor

# 모델: tiny/base/small/medium/large-v3
MODEL = "small"
# device="auto"면 GPU 있으면 GPU로, 없으면 CPU
asr = WhisperModel(MODEL, device="auto", compute_type="auto")

def transcribe(path: str):
    segments, info = asr.transcribe(path, language="ko", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))
    text = "".join(seg.text for seg in segments).strip()
    floor = parse_floor(text)
    # print({"text": text, "floor": floor})
    return {"text": text, "floor": floor}

if __name__ == "__main__":
    # 기본 샘플 경로 (현재 파일 위치 기준)
    default_voice = os.path.join(os.path.dirname(__file__), "./Test_data/sample.m4a")

    if len(sys.argv) < 2:
        print(f"[INFO] 입력 인자 없음 → 기본 샘플 사용: {default_voice}")
        voice = default_voice
    else:
        voice = sys.argv[1]

    transcribe(voice)
