# webcam_capture_focus.py
import cv2
import datetime

WIN = "webcam"

def apply_camera_settings(cap, w=1920, h=1080, fps=30, autofocus=False, focus_val=30):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 초점 제어 (장치 지원 시)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)

    # 실제 적용값 로그
    actual = {
        "W": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "H": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "FPS": cap.get(cv2.CAP_PROP_FPS),
        "AUTOFOCUS": int(cap.get(cv2.CAP_PROP_AUTOFOCUS)) if cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1 else -1,
        "FOCUS": cap.get(cv2.CAP_PROP_FOCUS) if cap.get(cv2.CAP_PROP_FOCUS) != -1 else -1,
    }
    print("[카메라] 적용:", actual)
    return actual

def main():
    cap = cv2.VideoCapture(0)  # 장치 인덱스 필요시 1,2로 변경
    if not cap.isOpened():
        raise RuntimeError("웹캠 열기 실패")

    autofocus = False      # 시작은 수동 포커스
    focus_val = 30
    apply_camera_settings(cap, autofocus=autofocus, focus_val=focus_val)

    print("키 안내 → Space: 캡처 / A: AF 토글 / [: 포커스- / ]: 포커스+ / ESC: 종료")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        # 안내 오버레이
        overlay = frame.copy()
        txt = f"AF={'ON' if autofocus else 'OFF'} | FOCUS={int(cap.get(cv2.CAP_PROP_FOCUS)) if cap.get(cv2.CAP_PROP_FOCUS)!=-1 else 'N/A'}"
        cv2.putText(overlay, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(WIN, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Space: 캡처
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{ts}.jpg"
            cv2.imwrite(filename, frame)
            print("saved:", filename)
        elif key in (ord('a'), ord('A')):  # AF 토글
            autofocus = not autofocus
            ret = cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            val = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            print(f"[AF] 토글 요청: {ret}, 현재값: {val}")
        elif key == ord('['):  # 포커스 -
            if not autofocus:
                focus_val = cap.get(cv2.CAP_PROP_FOCUS)
                if focus_val == -1:
                    print("이 장치는 수동 포커스 미지원")
                else:
                    focus_val = max(0, focus_val - 2)
                    ret = cap.set(cv2.CAP_PROP_FOCUS, focus_val)
                    val = cap.get(cv2.CAP_PROP_FOCUS)
                    print(f"[FOCUS-] 요청: {ret}, 현재값: {val}")
            else:
                print("AF ON 상태에선 수동 조정 불가")
        elif key == ord(']'):  # 포커스 +
            if not autofocus:
                focus_val = cap.get(cv2.CAP_PROP_FOCUS)
                if focus_val == -1:
                    print("이 장치는 수동 포커스 미지원")
                else:
                    focus_val = min(255, focus_val + 2)
                    ret = cap.set(cv2.CAP_PROP_FOCUS, focus_val)
                    val = cap.get(cv2.CAP_PROP_FOCUS)
                    print(f"[FOCUS+] 요청: {ret}, 현재값: {val}")
            else:
                print("AF ON 상태에선 수동 조정 불가")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
