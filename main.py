from Speech_recognition import transcribe_file_faster
from Scene_recognition.Elevator_OCR_RCNN_V2 import find_buttons as fb
from Camera import depth_from_shift
import util
import time

CTX = fb.preload_models(
    det_path="Scene_recognition/Elevator_OCR_RCNN_V2/frozen_model/detection_graph.pb",
    ocr_path="Scene_recognition/Elevator_OCR_RCNN_V2/frozen_model/ocr_graph.pb",
    do_warmup=True
)
print("\n모델 로드 완료\n")

VOICE_PATH = "Speech_recognition/Test_data/sample.m4a"
IMAGE_PATH_FOR_TEST_1 = "Scene_recognition/Test_data/sample_glossy_paper_vertical.png"
IMAGE_PATH_FOR_TEST_2 = "Scene_recognition/Test_data/sample_glossy_paper_vertical2.png"

# 음성 인식
print("음성 인식 중...")
speech_to_text = transcribe_file_faster.transcribe(VOICE_PATH)
print(speech_to_text)
print(f"target: {speech_to_text['floor']}\n")

# 첫 패널 촬영 
buttons_raw = fb.find_buttons(['--image', IMAGE_PATH_FOR_TEST_1, '--output', 'Image_backup/step0.png', '--no_vis'], ctx=CTX)

prev_offset = None
i = 1

while True:
    center = util.find_center_for_distance(fb, IMAGE_PATH_FOR_TEST_1, str(speech_to_text["floor"]), CTX, out_name=f"step1-{i}.png")
    i += 1
    # 중앙 판정
    done, offset, cmd = util.is_centered(center, prev_offset, img_size=(1920,1080), tol_px=10)

    print("offset:", offset, "cmd:", cmd)

    if done:
        print("중앙에 위치 ✓")
        break

    # 모터 제어
    if cmd == "left":
        print("왼쪽으로 1도 회전")
        # motor.rotate(-1)   # 왼쪽으로 1도
    elif cmd == "right":
        print("오른쪽으로 1도 회전")
        # motor.rotate(1)    # 오른쪽으로 1도

    prev_offset = offset
    time.sleep(2)
    break

# 거리 측정용 첫번째 패널 촬영
first_center = util.find_center_for_distance(fb, IMAGE_PATH_FOR_TEST_1, str(speech_to_text["floor"]), CTX, out_name="step2.png")

# 로봇이 위나 아래로 움직여야 함 거리측정용 (50mm)

# 이동후 두번째 패널 촬영
second_center = util.find_center_for_distance(fb, IMAGE_PATH_FOR_TEST_2, str(speech_to_text["floor"]), CTX, out_name="step3.png")

print("첫 번째 버튼 중심 좌표:", first_center)
print("두 번째 버튼 중심 좌표:", second_center)

# 깊이 추정 (단위: mm)
depth = depth_from_shift.depth_from_vertical_shift(50, first_center, second_center)
print(f"추정 깊이 Z ≈ {depth:.2f} mm")

# 팔 움직이기