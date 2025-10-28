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
from IK_FK import enhanced_plot, find_target_in_txt
import util
from Speech_recognition import transcribe_file_faster  # 필요하면 주석 해제

from Robot.RobotArm import RobotArm

arm = RobotArm("/dev/cu.usbmodem1101",
                baudrate=9600,
                map_us_min=500, map_us_max=2500,   # 매핑 범위(실서보)
                safe_us_min=644, safe_us_max=2300) # 안전(제한) 범위

arm.set_offset(2, +10)
arm.set_offset(3, -10)
arm.set_offset(4, +6)

arm.set_reversed(2, True)
arm.set_reversed(4, True)

arm.set_angle(5, 70)  # 베이스 회전 초기 위치
arm.set_angle(4, 128)
arm.set_angle(3, 74)
arm.set_angle(2, 68)

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

from queue import Queue

# ---- 전역 상태 ----
task_q = Queue(maxsize=1)  # 동시에 1건만 처리

def worker():
    while True:
        frame_bgr, floor_str = task_q.get()
        try:
            run_detection_async(frame_bgr, floor_str)
        except Exception as e:
            print("[워커] 처리 중 오류:", e)
        finally:
            task_q.task_done()

# 프로그램 시작 시 워커 하나만 띄우기 (while 루프 들어가기 전에)
threading.Thread(target=worker, daemon=True).start()

# --- 상태 ---
prev_offset = None
is_busy = False
last_center = None          # 마지막으로 찾은 중심 좌표
last_cmd = None             # last is_centered 결과
floor_target = None         # 음성인식 사용 시 타겟층 저장

# 음성 인식 사용하려면:
#VOICE_PATH = "Speech_recognition/Test_data/sample.m4a"
#stt = transcribe_file_faster.transcribe(VOICE_PATH)
#floor_target = str(stt.get("floor"))
#print("음성 인식:", stt)

floor_target = 3

def run_detection_async(frame_bgr, floor_str):
    global prev_offset, last_center, last_cmd

    try:
        snap_name = datetime.datetime.now().strftime("%H%M%S_%f")
        snap_path = f"tmp/snap_{snap_name}.png"
        out_vis_path = f"Image_backup/step0_{snap_name}.png"
        out_center_vis = f"Image_backup/step1_{snap_name}.png"

        cv2.imwrite(snap_path, frame_bgr)
        _ = fb.find_buttons(['--image', snap_path, '--output', out_vis_path, '--no_vis'], ctx=CTX)

        center = util.find_center_for_distance(fb, snap_path, str(floor_str), CTX, out_name=out_center_vis)

        done, offset, cmd = util.is_centered(center, img_size=(W, H), tol_px=70)
        last_center = center
        last_cmd = (done, offset, cmd)

        print(f"[탐지] center={center}, done={done}, offset={offset}, cmd={cmd}")

        if done:
            print("중앙에 위치 ✓")
            prev_offset = offset
            return

        if cmd == "left":
            print("왼쪽으로 1도 회전")
            cur = arm._us_to_deg_map(arm._last_us[5])
            arm.set_angle(5, cur + 1)
            prev_offset = offset
            return

        if cmd == "right":
            print("오른쪽으로 1도 회전(샘플)")
            cur = arm._us_to_deg_map(arm._last_us[5])
            arm.set_angle(5, cur - 1)
            prev_offset = offset
            return

    except Exception as e:
        print("[오류] 탐지 실패:", e)


print("스페이스바: 버튼 탐지 / Enter: 현재 중심 재표시 / m: 깊이 1샷 / n: 깊이 2샷 후 계산 / g: 버튼 누르기 / ESC: 종료")

depth_sample_1 = None  # (스냅파일경로, center)
depth_sample_2 = None  # (스냅파일경로, center)

def get_fresh_frame(cap, warmup=3):
    # 버퍼에 남은 오래된 프레임 비우기 + 최신 프레임 확보
    for _ in range(max(0, warmup-1)):
        cap.grab()               # 읽지 않고 버리기
    ok, frm = cap.read()
    if not ok:
        raise RuntimeError("프레임 캡처 실패")
    return frm

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

    elif key == 32:  # Space
        if task_q.full():
            print("탐지 중... (대기 큐가 가득)")
        else:
            # 현재 프레임 복사본을 큐에 넣고, 워커 1개가 처리
            task_q.put((frame.copy(), floor_target))
            print("탐지 요청 등록")

    elif key == 13:  # Enter: 마지막 중심 콘솔 재로그
        print("[상태] last_center:", last_center, " last_cmd:", last_cmd)

    # ── m 키: 첫 샷 ───────────────────────────────────────────────
    elif key == ord('m'):
        arm.set_angle(4, 128)
        arm.set_angle(3, 74)
        arm.set_angle(2, 68)
        time.sleep(0.6)  # 너무 길 필요 없음. 대신 프레임 몇 장 버려주자.

        c1 = None
        for attempt in range(20):
            # 최신 프레임으로 갱신! (여기가 핵심)
            frame_now = get_fresh_frame(cap, warmup=3)

            snap1 = f"tmp/depth1_{int(time.time())}_{attempt+1}.png"
            cv2.imwrite(snap1, frame_now)

            try:
                c1 = util.find_center_for_distance(
                    fb, snap1, str(floor_target), CTX,
                    out_name=f"depth_step1_try{attempt+1}.png"
                )
                if c1 is not None and len(c1) == 2:
                    depth_sample_1 = (snap1, c1)
                    print(f"[깊이] 첫 번째 샷 성공 (시도 {attempt+1}/20):", snap1, "center:", c1)
                    print("이제 로봇을 수직으로 50mm 이동하고 'n'을 눌러 두 번째 샷을 찍으세요.")
                    break
                else:
                    print(f"[깊이] 탐지 실패 (시도 {attempt+1}/20): center={c1}")
            except Exception as e:
                print(f"[깊이] 에러 발생 (시도 {attempt+1}/20): {e}")
            time.sleep(0.2)  # 너무 길게 기다릴 필요 없음

        if c1 is None:
            print("[깊이] 첫 번째 샷 20회 시도했지만 실패")

    # ── n 키: 두 번째 샷 ─────────────────────────────────────────
    elif key == ord('n'):
        if depth_sample_1 is None:
            print("먼저 'm'으로 첫 샷을 찍어주세요.")
        else:
            arm.set_angle(2, 51)
            arm.set_angle(3, 105)
            arm.set_angle(4, 114)
            time.sleep(0.6)

            c2 = None
            for attempt in range(20):
                frame_now = get_fresh_frame(cap, warmup=3)

                snap2 = f"tmp/depth2_{int(time.time())}_{attempt+1}.png"
                cv2.imwrite(snap2, frame_now)

                try:
                    c2 = util.find_center_for_distance(
                        fb, snap2, str(floor_target), CTX,
                        out_name=f"depth_step2_try{attempt+1}.png"
                    )
                    if c2 is not None and len(c2) == 2:
                        print(f"[깊이] 두 번째 샷 성공 (시도 {attempt+1}/20):", snap2, "center:", c2)
                        break
                    else:
                        print(f"[깊이] 탐지 실패 (시도 {attempt+1}/20): center={c2}")
                except Exception as e:
                    print(f"[깊이] 에러 발생 (시도 {attempt+1}/20): {e}")
                time.sleep(0.2)

            if c2 is None:
                print("[깊이] 두 번째 샷 20회 시도했지만 실패")
            else:
                try:
                    z_mm = depth_from_shift.depth_from_vertical_shift(59, depth_sample_1[1], c2)
                    print(f"[깊이] 추정 Z ≈ {z_mm:.2f} mm")
                except Exception as e:
                    print("[깊이] 계산 실패:", e)
                finally:
                    depth_sample_1 = None

    elif key == ord('g'):  # Go: 현재 깊이와 중심으로 이동 명령
        if depth_sample_2 is None:
            print("[MOVE] 깊이 정보 없음. m→n 순서로 깊이 먼저 계산하세요.")
        else:
            print("[MOVE] 버튼 누르기 동작 실행 (샘플)")
            current_x, current_y = enhanced_plot.end_effector_xy(70, 120, 40, L=(104,145,180), base=(0,92), mode="ui")
            add_depth_x = current_x - z_mm
            new_coords = (add_depth_x, current_y)

            find_target_in_txt.find_target_in_file(new_coords, "./IK_FK/angles_coords_step1.txt")

