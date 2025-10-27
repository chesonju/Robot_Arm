# -*- coding: utf-8 -*-
import cv2
import os
import time
import datetime
import threading
import numpy as np
from pathlib import Path

from Scene_recognition.Elevator_OCR_RCNN_V2 import find_buttons as fb
from Camera import depth_from_shift
import util
from Speech_recognition import transcribe_file_faster  # 필요하면 주석 해제

from Robot import RobotArm

arm = RobotArm("/dev/ttyUSB0", 9600, us_min=544, us_max=2400)

motor1_Value = 90  # 초기 각도 설정
motor2_Value = 90  # 초기 각도 설정
motor3_Value = 90  # 초기 각도 설정
motor4_Value = 90  # 초기 각도 설정
motor5_Value = 90  # 초기 각도 설정
motor6_Value = 90  # 초기 각도 설정
motor7_Value = 90  # 초기 각도 설정

arm.set_angles({
    1: motor1_Value,
    2: motor2_Value,
    3: motor3_Value,
    4: motor4_Value,
    5: motor5_Value,
    6: motor6_Value,
    7: motor7_Value
    })      

# --- 준비: 디렉토리 ---
Path("recordings").mkdir(exist_ok=True, parents=True)
Path("tmp").mkdir(exist_ok=True, parents=True)
Path("Image_backup").mkdir(exist_ok=True, parents=True)

# --- 모델 프리로드 ---
CTX = fb.preload_models(
    det_path="Scene_recognition/Elevator_OCR_RCNN_V2/frozen_model/detection_graph.pb",
    ocr_path="Scene_recognition/Elevator_OCR_RCNN_V2/frozen_model/ocr_graph.pb",
    do_warmup=True
)
print("\n모델 로드 완료\n")

# --- 웹캠 오픈 ---
cap = cv2.VideoCapture(0)  # 필요 시 1,2로 바꿔
if not cap.isOpened():
    raise RuntimeError("웹캠 오픈 실패")

# 해상도/FPS 설정 (장치 따라 무시될 수 있음)
W, H, FPS = 1920, 1080, 30
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)

# 실제 적용된 사이즈 읽기
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or W
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or H
FPS = cap.get(cv2.CAP_PROP_FPS) or FPS

# --- 레코더(항상 녹화) ---
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

rec_path = f"recordings/cam_{ts}.mp4"
rec_overlay_path = f"recordings/overlay_{ts}.mp4"

rec_raw = cv2.VideoWriter(rec_path, fourcc, FPS, (W, H))
rec_overlay = cv2.VideoWriter(rec_overlay_path, fourcc, FPS, (W, H))
print(f"[녹화 시작] {rec_path}")

# --- 상태 ---
prev_offset = None
is_busy = False
last_center = None          # 마지막으로 찾은 중심 좌표
last_cmd = None             # last is_centered 결과
floor_target = None         # 음성인식 사용 시 타겟층 저장

# 음성 인식 사용하려면:
VOICE_PATH = "Speech_recognition/Test_data/sample.m4a"
stt = transcribe_file_faster.transcribe(VOICE_PATH)
floor_target = str(stt.get("floor"))
print("음성 인식:", stt)

# === 너의 캘리브레이션 결과 ===
K_calib = np.array([[304.92065479,   0.,           156.23274445],
                    [  0.,           313.68343277, 126.36038472],
                    [  0.,             0.,           1.        ]], dtype=np.float64)

dist_calib = np.array([[-9.67711486e-03, -2.24344024e-01, -1.53467994e-03, 
                         6.54834768e-04,  1.48537539e+00]], dtype=np.float64)  # k1,k2,p1,p2,k3

# 캘리브 당시 해상도(추정). cx,cy를 보면 320x240 느낌이라 이렇게 둔다.
CALIB_W, CALIB_H = 320, 240  # 실제 네가 캘리브 때 썼던 사이즈로 바꿔도 됨

def scale_intrinsics(K_src: np.ndarray, src_size: tuple[int,int], dst_size: tuple[int,int]) -> np.ndarray:
    """해상도가 달라졌을 때 K 스케일링. (간단 스케일; 리프로젝션 최적은 아님)"""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / src_w
    sy = dst_h / src_h
    K = K_src.copy()
    K[0,0] *= sx  # fx
    K[1,1] *= sy  # fy
    K[0,2] *= sx  # cx
    K[1,2] *= sy  # cy
    return K

# 웹캠 실제 사이즈 W,H 읽은 다음에:
K = scale_intrinsics(K_calib, (CALIB_W, CALIB_H), (W, H))
dist = dist_calib  # 왜곡계수는 보통 그대로 사용(스케일 영향 적음)

def pixel_to_camera_xyz_mm(u, v, Z_mm, K: np.ndarray, dist: np.ndarray):
    """
    왜곡 포함 픽셀좌표(u,v) + 깊이 Z(mm) → 카메라 좌표계(mm).
    카메라 좌표계: 오른쪽+X, 아래+Y, 앞+Z 가정.
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)  # shape (1,1,2)
    # 정규화 좌표 (왜곡 제거 + K 반영)
    undist = cv2.undistortPoints(pts, K, dist, P=None)  # shape (1,1,2), (x_n, y_n)
    x_n = undist[0,0,0]
    y_n = undist[0,0,1]
    X = x_n * Z_mm
    Y = y_n * Z_mm
    Z = Z_mm
    return X, Y, Z

def run_detection_async(frame_bgr, floor_str):
    global is_busy, prev_offset, last_center, last_cmd
    try:
        snap_name = datetime.datetime.now().strftime("%H%M%S_%f")
        snap_path = f"tmp/snap_{snap_name}.png"
        out_vis_path = f"Image_backup/step0_{snap_name}.png"
        out_center_vis = f"Image_backup/step1_{snap_name}.png"

        # 프레임 저장
        cv2.imwrite(snap_path, frame_bgr)

        # 버튼 검출(파일 경로 인자 기반)
        _ = fb.find_buttons(['--image', snap_path, '--output', out_vis_path, '--no_vis'], ctx=CTX)

        # 중심 계산 & 중앙 판정
        center = util.find_center_for_distance(fb, snap_path, str(floor_str), CTX, out_name=out_center_vis)

        done, offset, cmd = util.is_centered(center, img_size=(W, H), tol_px=70)
        last_center = center
        last_cmd = (done, offset, cmd)

        # 로그
        print(f"[탐지] center={center}, done={done}, offset={offset}, cmd={cmd}")
        if done:
            print("중앙에 위치 ✓")
        else:
            if cmd == "left":
                print("왼쪽으로 1도 회전")
                arm.set_angle(0, motor1_Value - 1)  # 1도 왼쪽 회전
            elif cmd == "right":
                print("오른쪽으로 1도 회전(샘플)")
                # motor.rotate(+1)
        prev_offset = offset

    except Exception as e:
        print("[오류] 탐지 실패:", e)
    finally:
        is_busy = False

print("스페이스바: 버튼 탐지 / Enter: 현재 중심 재표시 / m: 깊이 1샷 / n: 깊이 2샷 후 계산 / g: 버튼 누르기 / ESC: 종료")

depth_sample_1 = None  # (스냅파일경로, center)
depth_sample_2 = None  # (스냅파일경로, center)

while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임 읽기 실패")
        break

    # 항상 녹화
    rec_raw.write(frame)

    # 미리보기용 복사본
    view = frame.copy()
    H, W = view.shape[:2]

    # 화면 중앙 x
    cx_mid = W // 2

    # 중앙 100픽셀 폭 → 좌우 경계선
    left_x = cx_mid - 50
    right_x = cx_mid + 50

    # 경계선은 view 위에 그릴 것!
    cv2.line(view, (left_x, 0),   (left_x, H), (0, 0, 255), 2)  # 빨강
    cv2.line(view, (right_x, 0),  (right_x, H), (0, 0, 255), 2)
    cv2.line(view, (cx_mid, 0),   (cx_mid, H), (0, 255, 0), 1)  # 중앙선(초록)

    # 마지막 검출 결과 오버레이
    if last_center:
        try:
            cx_btn, cy_btn = int(last_center[0]), int(last_center[1])
            cv2.circle(view, (cx_btn, cy_btn), 12, (0, 255, 0), 2)
            cv2.putText(view, f"center=({cx_btn},{cy_btn})", (cx_btn+10, cy_btn-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

    if last_cmd:
        done, offset, cmd = last_cmd
        cv2.putText(view, f"done={done} offset={offset} cmd={cmd}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(view, f"REC {datetime.datetime.now().strftime('%H:%M:%S')}", (20, H-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    rec_overlay.write(view)
    cv2.imshow("webcam", view)
    key = cv2.waitKey(1) & 0xFF


    if key == 27:  # ESC
        print("종료")
        break

    elif key == 32:  # Space: 현재 프레임으로 버튼 탐지
        if not is_busy:
            is_busy = True
            # 복사본 넘겨서 안전하게 처리
            threading.Thread(target=run_detection_async, args=(frame.copy(), floor_target), daemon=True).start()
        else:
            print("탐지 중...")

    elif key == 13:  # Enter: 마지막 중심 콘솔 재로그
        print("[상태] last_center:", last_center, " last_cmd:", last_cmd)

    elif key == ord('m'):  # 깊이 추정 1샷 (수직으로 50mm 이동 전)
        snap1 = f"tmp/depth1_{int(time.time())}.png"
        cv2.imwrite(snap1, frame)
        c1 = util.find_center_for_distance(fb, snap1, str(floor_target), CTX, out_name="depth_step1.png")
        depth_sample_1 = (snap1, c1)
        print("[깊이] 첫 번째 샷 저장:", snap1, "center:", c1)
        print("이제 로봇을 수직으로 50mm 이동하고 'n'을 눌러 두 번째 샷을 찍으세요.")

    elif key == ord('n'):  # 깊이 추정 2샷 (수직으로 50mm 이동 후)
        if depth_sample_1 is None:
            print("먼저 'm'으로 첫 샷을 찍어주세요.")
        else:
            snap2 = f"tmp/depth2_{int(time.time())}.png"
            cv2.imwrite(snap2, frame)
            c2 = util.find_center_for_distance(fb, snap2, str(floor_target), CTX, out_name="depth_step2.png")
            print("[깊이] 두 번째 샷 저장:", snap2, "center:", c2)
            try:
                z_mm = depth_from_shift.depth_from_vertical_shift(50, depth_sample_1[1], c2)
                print(f"[깊이] 추정 Z ≈ {z_mm:.2f} mm")
            except Exception as e:
                print("[깊이] 계산 실패:", e)
            finally:
                depth_sample_1 = None

    elif key == ord('g'):  # Go: 현재 깊이와 중심으로 이동 명령
        if depth_sample_2 is None:
            print("[MOVE] 깊이 정보 없음. m→n 순서로 깊이 먼저 계산하세요.")
        else:
            _, c2, z_mm = depth_sample_2
            u, v = float(c2[0]), float(c2[1])

            try:
                X_mm, Y_mm, Z_mm = pixel_to_camera_xyz_mm(u, v, z_mm, K, dist)
                print(f"[MOVE] Pixel({u:.1f},{v:.1f}) + Z={Z_mm:.1f} → Camera XYZ(mm)=({X_mm:.1f}, {Y_mm:.1f}, {Z_mm:.1f})")
                move_to_xyz_mm(X_mm, Y_mm, Z_mm)  # 여기에 실제 로봇 제어
            except Exception as e:
                print("[MOVE] 변환 실패:", e)


