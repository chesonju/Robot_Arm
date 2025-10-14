# transcribe_file_whisper.py
import sys
import os
import whisper

try:
    from Speech_recognition.parse_floor import parse_floor
except ImportError:
    from parse_floor import parse_floor

# 모델 사이즈: tiny/base/small/medium/large-v3
MODEL = "small"  # GPU 있으면 "medium"이나 "large-v3"도 가능
model = whisper.load_model(MODEL)

def transcribe(path: str):
    # 한국어로 강제 인식하고 싶으면 language="ko" 지정
    result = model.transcribe(path, language="ko", fp16=False)
    text = result.get("text", "").strip()
    floor = parse_floor(text)
    print({"text": text, "floor": floor})

if __name__ == "__main__":
    # 기본 샘플 경로 (현재 파일 위치 기준)
    default_voice = os.path.join(os.path.dirname(__file__), ".Test_data/sample.m4a")

    if len(sys.argv) < 2:
        print(f"[INFO] 입력 인자 없음 → 기본 샘플 사용: {default_voice}")
        voice = default_voice
    else:
        voice = sys.argv[1]

    transcribe(voice)