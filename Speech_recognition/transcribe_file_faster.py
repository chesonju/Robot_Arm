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

def is_yes_no(text: str) -> str:
    """
    입력 텍스트가 'yes' 또는 'no'인지 판단합니다.
    (대소문자를 구분하지 않고 앞뒤 공백을 제거합니다.)

    :param text: transcribe 함수에서 반환된 텍스트
    :return: "yes", "no", 또는 "other"
    """
    clean_text = text.strip().lower()
    
    # 한국어 "예/아니오"와 영어 "yes/no"를 함께 처리
    if clean_text in ["yes", "네", "예", "응"]:
        return "yes"
    elif clean_text in ["no", "아니오", "아니", "놉"]:
        return "no"
    else:
        return "other"
    
def correction_command(text: str) -> str | None:
    """
    로봇팔의 핵심 명령어("위로", "아래로", "앞으로")를 구분합니다.

    :param text: Whisper로 변환된 텍스트
    :return: "up", "down", "forward" 중 하나 또는 None
    """
    # 텍스트 전처리: 앞뒤 공백 제거 및 소문자 변환 후 공백 제거
    clean_text = text.strip().lower().replace(" ", "")

    if clean_text.startswith("위로"):
        return "up"
    elif clean_text.startswith("아래로"):
        return "down"
    elif clean_text.startswith("앞으로"):
        return "forward"
    else:
        return None

def transcribe(path: str):
    segments, info = asr.transcribe(path, language="ko", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))
    text = "".join(seg.text for seg in segments).strip()
    floor = parse_floor(text)
    is_yes_or_no = is_yes_no(text)
    correction = correction_command(text)

    result = {
        "text": text, 
        "floor": floor, 
        "is_yes_no": is_yes_or_no,
        "correction": correction
    }
    print(result)

    return result

if __name__ == "__main__":
    # 기본 샘플 경로 (현재 파일 위치 기준)
    default_voice = os.path.join(os.path.dirname(__file__), "./Test_data/sample.m4a")

    if len(sys.argv) < 2:
        print(f"[INFO] 입력 인자 없음 → 기본 샘플 사용: {default_voice}")
        voice = default_voice
    else:
        voice = sys.argv[1]

    transcribe(voice)
