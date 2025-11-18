# wake_word_detector.py 파일 (수정됨: 현재 파일 위치 기준으로 경로 설정)

from openwakeword.model import Model
import pyaudio
import numpy as np
import time
import sys
import os
import chime
import time
import os
import wave
import pyaudio
import numpy as np
from . import transcribe_file_faster

# 현재 스크립트 파일이 위치한 디렉터리 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_wake_word(keyword: str, threshold: float = 0.5) -> bool:
    chime.info()
    # 1. 모델 경로를 절대 경로로 설정
    # os.path.join()을 사용하여 운영체제에 관계없이 올바른 경로를 구성합니다.
    model_dir = os.path.join(CURRENT_DIR, "wake_word_detector_model")
    
    if keyword == "ok computer":
        tflite_model_path = os.path.join(model_dir, "ok_computer.tflite")
    elif keyword == "hey luna":
        tflite_model_path = os.path.join(model_dir, "hey_luna.tflite")
    else:
        raise Exception(f"해당 wwd model 없음: {keyword}")
    
    # ... (Model 로딩 코드는 그대로 유지) ...
    try:
        if not os.path.exists(tflite_model_path):
            # 파일 경로 오류 시, 찾고 있는 경로를 명확히 출력하여 디버깅을 돕습니다.
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {tflite_model_path}")

        # 1. Model 인스턴스 생성 (직접 다운로드한 모델 경로 사용)
        oww_model = Model(wakeword_models=[tflite_model_path])
        print(f"✅ openwakeword 모델 로딩 완료: {keyword}")
        
    except Exception as e:
        print(f"openwakeword 모델 초기화 오류: {e}")
        print(f"모델 파일 경로: {tflite_model_path} 가 정확한지 확인해 주세요.")
        sys.exit(1)
        
    # ... (나머지 detect_wake_word 함수 코드는 그대로 유지) ...
    
    if oww_model is None:
        return False
        
    # --- PyAudio 설정 ---
    CHUNK = 1280
    FORMAT = pyaudio.paInt16 
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = None

    try:
        stream = p.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK)

        print(f"🎤 상시 대기 시작! (호출어: {keyword}, 임계값: {threshold})")
        print("종료하려면 Ctrl+C를 누르세요.")

        while True:
            # 1. 오디오 데이터 읽기
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)
            
            # 2. Model 인스턴스의 predict 함수 사용
            predictions = oww_model.predict(audio_frame)

            # 3. 키워드 감지 확인 및 반환
            file_name = os.path.basename(tflite_model_path)
            model_key = os.path.splitext(file_name)[0]
            
            if model_key in predictions and predictions[model_key] > threshold:
                chime.success()
                print("\n-------------------------------------------")
                print(f"✨ **웨이크 워드 '{keyword}' 감지됨!** ✨")
                print("-------------------------------------------\n")
                return True

    except KeyboardInterrupt:
        print("\n웨이크 워드 감지 중지.")
        return False
        
    except Exception as e:
        print(f"오디오 또는 감지 중 오류 발생: {e}")
        return False

    finally:
        # 리소스 정리
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()

# record_command 함수는 변경 없이 그대로 사용합니다.
def record_command(duration_seconds: int = 3, filename: str = "temp_recorded_file"):
    """웨이크 워드 감지 후 사용자의 명령을 녹음하고 파일로 저장합니다."""
    # ... (기존 record_command 함수 코드 삽입) ...
    # PyAudio 설정 (wake_word_detector와 동일하게 16kHz 설정)
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  
    
    p = pyaudio.PyAudio()
    frames = []

    print(f"👂 명령 녹음 시작 ({duration_seconds}초)...")
    
    # 마이크 스트림 열기
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # 지정된 시간 동안 녹음
    for i in range(0, int(RATE / CHUNK * duration_seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("🛑 녹음 완료.")
    chime.success()

    # 스트림 및 PyAudio 정리
    stream.stop_stream()
    stream.close()
    p.terminate()

    # WAV 파일로 저장
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename


if __name__ == '__main__':
    chime.info()
    # 메인 테스트
    if detect_wake_word(keyword="ok computer", threshold=0.5):
        recorded_file = record_command(filename="Speech_recognition/temp_command_1.wav")
        transcription_result = transcribe_file_faster.transcribe(recorded_file)
        # print(f"STT 결과: {transcription_result['text']}")
        print(transcription_result)
        
    else:
        print("웨이크 워드 감지가 중지되었습니다.")

    print(f"text -> {transcription_result.get("text")}")
    print(f"floor -> {transcription_result.get("floor")}")
    print(f"is_yes_no -> {transcription_result.get("is_yes_no")}")
    print(f"correction -> {transcription_result.get("correction")}")