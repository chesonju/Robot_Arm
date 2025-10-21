# 실행

uv 환경 으로 구축했습니다

실행은 **루트폴더**에서 (ROBOT_ARM)

```bash
brew install uv
uv sync # <- 필요 패키지 설치

uv run main_cv.py
uv run main_cv.py 2>/dev/null
```

아래와 같이 실행 하면 Scene_recognition/Test_data/Sample.png 를 사용하고Robot_Arm/Scene_recognition/Elevator_OCR_RCNN_V2/OCR_RCNN_V2_out.png 에 결과파일이 나옵니다

(별도 인자로 이미지 경로 설정 가능하고 main.py에서 그렇게 씁니다)

```bash
uv run Scene_recognition/Elevator_OCR_RCNN_V2/find_buttons.py 2>/dev/null
```

# ROBOT ARM

**거리 관련 기술 필요**

## 폴더 설명

- Camera
  - 카메라 캘리브레이션
- IK_FK
  - 역기구학, 정기구학 연산
- img_bak
  - 자료 이미지 모음
- Scene_recognition
  - 장면 인식 (현재 엘리베이터)
- Speech_recognition
  - 음성 데이터 -> string으로 변환 후 처리 (ex "3층 눌러줘")

## 자료

### STT 모델

1. https://github.com/openai/whisper

### 엘리베이터 버튼 인식 모델

(dataset 있음)

1. https://github.com/zhudelong/elevator_button_recognition
2. https://www.researchgate.net/figure/A-demonstration-of-button-recognition-result-Black-and-green-bounding-boxes-indicate-the_fig1_330586242
3. https://github.com/zhudelong/ocr-rcnn-v2?tab=readme-ov-file

