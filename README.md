# 해야 할 것

하드웨어

* [X]  하드웨어 조립 (로봇 프레임 서보 브라켓 이용)
* [X]  로봇 팔 측 아두이노 코드 작성
* [X]  데모용 엘리베이터 패널 제작
* [ ]  웹캠 캘리브레이션 (chessboard 50장 정도는 있어야 함)
* [ ]  캘리브레이션 결과 (초점거리 등) 로 거리 측정 테스트

소프트웨어

* [X]  엘리베이터 버튼 인식 환경 구축 (ocr -> RCNN)
* [X]  IK_FK 구축  (IK 는 너무 변수가 많아서 현재 FK테이블 사용)
* [X]  Speech To Text  (타겟 층수를 음성에서 추출)
* [X]  카메라에 잡힌 영상 상 버튼 위치 판독

통합

* [ ]  아두이노 ↔️ IK_FK 간 규격 통일 코드 작성 (IK_FK는 각도기반)
* [ ]  버튼 위치에 따라 로봇 팔 회전으로 중심 정렬
* [ ]  모터의 이상적인 각도 대비 오차 보정
* [ ]  IK_FK/calc_all_angles.py 로 모든 계산 (find_target_in_txt.py) 에서 사용
* [ ]  x만큼 이동한 pair 후보 중 골라서 해당 각도로 움직임 테스트 (거리측정용)
* [ ]  동작 영상 촬영
* [ ]  PPT 작성

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

IK_FK는 실행전 페어 계산이 필요합니다

~~~bash
uv run IK_FK/calc_all_angles.py # 모든 페어 계산

uv run IK_FK/find_target_in_txt.py # 거리 측정용 페어 계산 & 시각화
~~~

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
- Panel
  - 데모용 버튼 패널 코드

## 자료

### STT 모델

1. https://github.com/openai/whisper

### 엘리베이터 버튼 인식 모델

(dataset 있음)

1. https://github.com/zhudelong/elevator_button_recognition
2. https://www.researchgate.net/figure/A-demonstration-of-button-recognition-result-Black-and-green-bounding-boxes-indicate-the_fig1_330586242
3. https://github.com/zhudelong/ocr-rcnn-v2?tab=readme-ov-file
