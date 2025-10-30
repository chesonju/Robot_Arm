# -*- coding: utf-8 -*-
import cv2
import os
import time
import datetime
import threading
import numpy as np
from pathlib import Path
from queue import Queue

from Scene_recognition.Elevator_OCR_RCNN_V2 import find_buttons as fb
from Camera import depth_from_shift
from IK_FK import enhanced_plot, find_target_in_txt, find_target_with_camera_distance, find_push_form
import util
from Speech_recognition import transcribe_file_faster  # 필요하면 주석 해제

from Robot.RobotArm import RobotArm

arm = RobotArm("/dev/cu.usbmodem1101",
                baudrate=9600,
                map_us_min=500, map_us_max=2500,   # 매핑 범위(실서보)
                safe_us_min=644, safe_us_max=2300) # 안전(제한) 범위

arm.set_offset(2, +8)
arm.set_offset(3, -6)
arm.set_offset(4, +6)

arm.set_reversed(2, True)
arm.set_reversed(4, True)

distance_candidate_mm = {}
# 초기 위치
print("로봇 암 초기 위치로 이동")
arm.set_angle(4, 158)
arm.set_angle(3, 91)
arm.set_angle(2, 21)
arm.set_angle(5, 75)  # 베이스 회전 초기 위치

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
show_circle = True
show_angles = False 

# 음성 인식 사용하려면:
#VOICE_PATH = "Speech_recognition/Test_data/sample.m4a"
#stt = transcribe_file_faster.transcribe(VOICE_PATH)
#floor_target = str(stt.get("floor"))
#print("음성 인식:", stt)

floor_target = 4    

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

def draw_angle_overlay(view, fy=2239.0469, cy=513.1642, width=1920, height=1080):
    import math, cv2, numpy as np

    height, width = view.shape[:2]

    # 수직 화각 절반
    fov_half = math.atan2(height/2, fy)
    fov_half_deg = int(np.degrees(fov_half))

    # 각도 스텝 줄이기 (예: 1도 단위)
    for angle in range(-fov_half_deg, fov_half_deg+1, 1):
        theta = math.radians(angle)
        y = int(round(cy + math.tan(theta) * fy))

        if 0 <= y < height:
            if angle % 5 == 0:  # 5도마다 굵은 선
                thickness = 2
                color = (0, 255, 0) if angle == 0 else (0, 200, 0)
                cv2.putText(view, f"{angle:+d}°", (50, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:               # 나머지 1도 단위는 얇은 점선 느낌
                thickness = 1
                color = (100, 100, 100)

            cv2.line(view, (0, y), (width, y), color, thickness)

    return view

def estimate_distance_button_ring(
    frame_bgr,
    center,                 # (cx, cy)
    f_px=2239.0469,         # 보통 fy
    D_real_mm=25.0,         # 버튼 실제 지름
    Z_hint_mm=250.0,        # 대충 현재 거리 힌트
    roi_size=360,
    ring_w=6,               # 고리 두께(px)
    alpha=1.0,              # (바깥-안쪽) 대비 가중
    beta=0.15,              # r_hint 패널티 가중
    out_path="Image_backup/debug_button_ring.png"
):
    H, W = frame_bgr.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    # --- ROI ---
    x1, y1 = max(0, cx - roi_size//2), max(0, cy - roi_size//2)
    x2, y2 = min(W, cx + roi_size//2), min(H, cy + roi_size//2)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        print("[거리] ROI 비어있음"); return None

    cx_r, cy_r = cx - x1, cy - y1

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)   # 살짝 대비 업
    edges = cv2.Canny(gray, 80, 160)

    # 기대 반지름(px)
    r_hint = (f_px * D_real_mm) / (2.0 * max(1e-6, Z_hint_mm))
    r_min = max(8, int(r_hint * 0.6))
    r_max = min(int(min(roi.shape[:2]) * 0.48), int(r_hint * 1.6))
    if r_min >= r_max:
        r_min = max(8, int(r_hint * 0.5)); r_max = r_min + 20

    def annulus_mask(shape, cx, cy, r, w):
        """반지름 r, 두께 w의 고리 마스크"""
        h, w_img = shape[:2]
        mask_outer = np.zeros((h, w_img), np.uint8)
        mask_inner = np.zeros((h, w_img), np.uint8)
        r = max(1, int(round(r)))
        t = max(1, int(round(w)))
        cv2.circle(mask_outer, (int(cx), int(cy)), r + t, 255, thickness=-1)
        cv2.circle(mask_inner, (int(cx), int(cy)), r - t, 255, thickness=-1)
        mask = cv2.subtract(mask_outer, mask_inner)
        return mask

    best = None
    for r in range(r_min, r_max + 1):
        # 1) 원둘레 근처 엣지 평균
        m_ring = annulus_mask(edges.shape, cx_r, cy_r, r, ring_w)
        edge_on_ring = cv2.mean(edges, m_ring)[0]  # 0채널 평균

        # 2) 바깥/안쪽 밝기 대비
        m_out = annulus_mask(gray.shape, cx_r, cy_r, r + 2*ring_w, ring_w)
        m_in  = annulus_mask(gray.shape, cx_r, cy_r, r - 2*ring_w, ring_w)
        outside = cv2.mean(gray, m_out)[0]
        inside  = cv2.mean(gray, m_in)[0]
        contrast = outside - inside

        # 3) r_hint로 당기는 패널티
        penalty = abs(r - r_hint)

        score = edge_on_ring + alpha * contrast - beta * penalty
        if (best is None) or (score > best[0]):
            best = (score, r, edge_on_ring, contrast)

    if best is None:
        print("[거리] 반지름 탐색 실패"); return None

    _, r_best, eo, ct = best
    d_px = 2.0 * r_best
    Z_mm = (f_px * D_real_mm) / max(1e-6, d_px)

    # --- 파일로만 시각화 저장 ---
    vis = roi.copy()
    cv2.circle(vis, (cx_r, cy_r), int(r_best), (0,255,0), 2)
    cv2.circle(vis, (cx_r, cy_r), 2, (0,0,255), 2)
    cv2.putText(vis, f"d={d_px:.1f}px  Z={Z_mm:.1f}mm", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(vis, f"r_hint~{r_hint:.1f}px  score={best[0]:.1f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    # out_path 안전 저장
    dname = os.path.dirname(out_path)
    if dname:
        os.makedirs(dname, exist_ok=True)
    cv2.imwrite(out_path, vis)

    print(f"[거리] d≈{d_px:.1f}px  Z≈{Z_mm:.1f}mm  (saved {out_path})")
    return Z_mm

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
    if last_center and show_circle:
        try:
            cx_btn, cy_btn = int(last_center[0]), int(last_center[1])

            # view와 같은 크기의 반투명 레이어 준비
            overlay = view.copy()
            # 원 내부를 색칠 (BGR, -1은 filled)
            cv2.circle(overlay, (cx_btn, cy_btn), 50, (0, 255, 0), -1)
            # overlay와 원본을 블렌딩 (alpha=0.3 정도 → 30%만 색 입히기)
            alpha = 0.5
            view = cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)

            # 원 테두리는 그대로 표시
            cv2.circle(view, (cx_btn, cy_btn), 50, (0, 255, 0), 2)
            cv2.putText(view, f"center=({cx_btn},{cy_btn})", (cx_btn+25, cy_btn-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        except Exception:
            pass

    if last_cmd:
        done, offset, cmd = last_cmd
        cv2.putText(view, f"done={done} offset={offset} cmd={cmd}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(view, f"REC {datetime.datetime.now().strftime('%H:%M:%S')}", (20, H-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if show_angles:
        view = draw_angle_overlay(view, fy=2239.0469, cy=513.1642, width=W, height=H)

    rec_overlay.write(view)
    cv2.imshow("webcam", view)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("종료")
        break

    elif key == 32:  # Space
        show_circle = True
        if task_q.full():
            print("탐지 중... (대기 큐가 가득)")
        else:
            # 현재 프레임 복사본을 큐에 넣고, 워커 1개가 처리
            task_q.put((frame.copy(), floor_target))
            print("탐지 요청 등록")

    elif key == 13:  # Enter: 마지막 중심 콘솔 재로그
        print("[상태] last_center:", last_center, " last_cmd:", last_cmd)

    elif key == ord("i"):
        arm.set_angle(4, 158)
        arm.set_angle(3, 91)
        arm.set_angle(2, 21)

    # ── m 키: 첫 샷 ───────────────────────────────────────────────
    elif key == ord('m'):
        show_angles = False
        show_circle = False
        arm.set_angle(4, 158)
        arm.set_angle(3, 91)
        arm.set_angle(2, 21)
        time.sleep(1)  # 너무 길 필요 없음. 대신 프레임 몇 장 버려주자.

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

                    try:
                        Z = estimate_distance_button_ring(
                        frame_now, c1,
                        Z_hint_mm=250.0,               # 대충 현재 추정 거리
                        out_path=f"debug/ring_depth_try{attempt+1}.png"
                        )
                        if Z is not None:
                            _, angle, _ = depth_from_shift.pixel_to_angles_with_undistort(c1)
                            end = enhanced_plot.end_effector_xy(128,74,68)
                            distance_candidate_mm["m_with_25mm"] = (Z, angle, end) 
                            print(f"[원기반 algorithm 1] Z ≈ {Z:.1f} mm")

                    except Exception as e:
                        print(f"[깊이] 거리추정 실패: {e}")

                    print("이제 로봇을 수직으로 이동하고 'n'을 눌러 두 번째 샷을 찍으세요.")
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
            arm.set_angle(2, 15)
            arm.set_angle(4, 142)
            arm.set_angle(3, 113)
            time.sleep(3)

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
                    z_mm = depth_from_shift.depth_from_vertical_shift(30, depth_sample_1[1], c2)
                    print(f"[깊이] 추정 Z ≈ {z_mm:.2f} mm")
                    _, angle, _ = depth_from_shift.pixel_to_angles_with_undistort(c1)
                    end = enhanced_plot.end_effector_xy(142,113,15)
                    distance_candidate_mm["n_with_triangulation"] = (z_mm, angle, end)

                    Z = estimate_distance_button_ring(
                    frame_now, c2,
                    Z_hint_mm=250.0,               # 대충 현재 추정 거리
                    out_path=f"debug/ring_depth_try{attempt+1}.png"
                    )
                    if Z is not None:
                        distance_candidate_mm["n_with_25mm"] = (Z, angle, end)
                        print(f"[원기반 algorithm 1] Z ≈ {Z:.1f} mm")

                    depth_sample_2 = (snap2, c2)

                except Exception as e:
                    print("[깊이] 계산 실패:", e)
                finally:
                    depth_sample_1 = None

    elif key == ord('g'):  # Go: 현재 깊이와 중심으로 이동 명령
        if depth_sample_2 is None:
            print("[MOVE] 깊이 정보 없음. m→n 순서로 깊이 먼저 계산하세요.")
        else:
            print("[MOVE] 버튼 누르기 동작 실행 (샘플)")
            cur_x1, cur_y1 = distance_candidate_mm["n_with_triangulation"][2]
            cur_x1 = round(cur_x1, 2)
            cur_y1 = round(cur_y1, 2)
            
            print(distance_candidate_mm)

            length = distance_candidate_mm["n_with_triangulation"][0]
            angle = distance_candidate_mm["n_with_triangulation"][1]
            end = distance_candidate_mm["n_with_triangulation"][2]

            target_coords = find_target_with_camera_distance.plot_geometry (
                x1=cur_x1, y1=cur_y1, length=length, angle_input=angle,
                show_plot=False,  
                save_path="plot_result_1.png"  # 저장할 파일 이름
            )

            print(f"target_coords -> {target_coords}")

            #result1 = find_target_in_txt.find_target_in_file(target_coords[0] + 20 , target_coords[1] - 32, "./IK_FK/angles_coords_step1.txt")
            #prefix = result1.split(" ")[0] 
            #coords = list(map(int, prefix.split("_")))

            result2 = find_push_form.find_and_run_target(target_coords[0] + 20 , target_coords[1] - 32)
            prefix = result2.split(" ")[0]
            coords = list(map(int, prefix.split("_")))

            arm.set_angle(2, int(coords[2]))
            arm.set_angle(3, int(coords[1]))
            time.sleep(5)
            arm.set_angle(4, int(coords[0]))

            input = input("반동 기동 확인 y n")

            if input == "y":
                move_back = int(coords[0]) - 10
                print(f"move back to {move_back}")
                if move_back > 170:
                    arm.set_angle(4, 158)
                    time.sleep(5)
                    arm.set_angle(4, int(coords[0]), for_push=True)
