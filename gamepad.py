import pygame, time
from Robot.RobotArm import RobotArm

pygame.init()
pygame.joystick.init()
pygame.display.set_mode((200, 100))

print("컨트롤러 대기 중... (10초 동안)")
js = None
t_end = time.time() + 10
while time.time() < t_end and js is None:
    for e in pygame.event.get():
        if e.type == pygame.JOYDEVICEADDED:
            idx = e.device_index
            js = pygame.joystick.Joystick(idx)
            js.init()
            print(f"[ADDED] idx={idx} name={js.get_name()}")
    if pygame.joystick.get_count() > 0 and js is None:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"[ENUM] name={js.get_name()}")
    time.sleep(0.1)

if js is None:
    raise SystemExit("10초 안에 컨트롤러가 안 잡힘")

num_axes = js.get_numaxes()
num_buttons = js.get_numbuttons()
print(f"axes: {num_axes} buttons: {num_buttons} hats: {js.get_numhats()}")

AXIS_THRESH = 0.9
IGNORE_NEGATIVE_AXES = {2, 5}  # LT, RT 트리거는 -1.0일 때 무시
STEP = 5

axis_state = {i: 0 for i in range(num_axes)}
button_state = {i: False for i in range(num_buttons)}

def dir_label(axis, sign):
    names = {
        0: ("LEFT", "RIGHT"),
        1: ("UP", "DOWN"),
        2: ("LT", "LT"),
        3: ("UP(RY-)", "DOWN(RY+)"),
        4: ("LEFT(RX-)", "RIGHT(RX+)"),
        5: ("LT2(-)", "RT(+)"),
    }
    neg, pos = names.get(axis, ("NEG", "POS"))
    return neg if sign < 0 else pos

def hat_label(value):
    x, y = value
    if   (x, y) == (0, 1):  return "↑ UP"
    elif (x, y) == (0, -1): return "↓ DOWN"
    elif (x, y) == (-1, 0): return "← LEFT"
    elif (x, y) == (1, 0):  return "→ RIGHT"
    elif (x, y) == (0, 0):  return "CENTER"
    else:                   return f"{value}"

print(f"[INFO] |value| ≥ {AXIS_THRESH} → 눌림 감지")
print("0.5초마다 전체 상태 갱신 출력\n")

arm = RobotArm("/dev/cu.usbmodem1101",
                baudrate=9600,
                map_us_min=500, map_us_max=2500,   # 매핑 범위(실서보)
                safe_us_min=644, safe_us_max=2300) # 안전(제한) 범위

# 모터 보정값 설정
arm.set_offset(2, +8)
arm.set_offset(3, -6)
arm.set_offset(4, +6)

arm.set_reversed(2, True)
arm.set_reversed(4, True)

ANGLE_5 = 90
ANGLE_4 = 90
ANGLE_3 = 90
ANGLE_2 = 90
ANGLE_1 = 90
ANGLE_0 = 30

# 초기 위치
arm.set_angle(5, ANGLE_5)
arm.set_angle(4, ANGLE_4)
arm.set_angle(3, ANGLE_3)
arm.set_angle(2, ANGLE_2)
arm.set_angle(1, ANGLE_1)
arm.set_angle(0, ANGLE_0) # 그리퍼

while True:
    for e in pygame.event.get():
        if e.type == pygame.JOYBUTTONDOWN:
            button_state[e.button] = True
        elif e.type == pygame.JOYBUTTONUP:
            button_state[e.button] = False
        elif e.type == pygame.JOYHATMOTION:
            print(f"HAT {e.hat}: {hat_label(e.value)}")
        elif e.type == pygame.JOYAXISMOTION:
            a = e.axis
            v = js.get_axis(a)
            # 트리거 축의 -1.0 기본값은 무시
            if a in IGNORE_NEGATIVE_AXES and v <= -0.99:
                axis_state[a] = 0
                continue
            if v >= AXIS_THRESH:
                axis_state[a] = +1
            elif v <= -AXIS_THRESH:
                axis_state[a] = -1
            else:
                axis_state[a] = 0

    print("=== 현재 입력 상태 ===")
    pressed_btns = [i for i, st in button_state.items() if st]
    if pressed_btns:
        print("버튼:", ", ".join(str(i) for i in pressed_btns))
        if 1 in pressed_btns:
            print("  → B이 눌렸습니다!")
            ANGLE_5 += STEP
            arm.set_angle(5, ANGLE_5)
        if 2 in pressed_btns:
            print("  → X이 눌렸습니다!")
            ANGLE_5 -= STEP
            arm.set_angle(5, ANGLE_5)
        if 4 in pressed_btns:
            ANGLE_0 -= STEP
            arm.set_angle(0, ANGLE_0)
            print("  → LB이 눌렸습니다!")
        if 5 in pressed_btns:
            ANGLE_0 += STEP
            arm.set_angle(0, ANGLE_0)
            print("  → RB이 눌렸습니다!")
    else:
        print("버튼: 없음")

    active_axes = []
    for i, s in axis_state.items():
        val = js.get_axis(i)
        # 트리거 기본값(-1.0)은 출력에서 완전히 제외
        if i in IGNORE_NEGATIVE_AXES and val <= -0.99:
            continue
        if s != 0:
            active_axes.append((i, s, val))

    if active_axes:
        for i, s, val in active_axes:
            if 1 == i and val < -0.9:
                ANGLE_3 -= STEP
                arm.set_angle(3, ANGLE_3)
                print("  → 왼쪽 스틱 위로!")
            if 1 == i and val > 0.9:
                ANGLE_3 += STEP
                arm.set_angle(3, ANGLE_3)
                print("  → 왼쪽 스틱 아래로!")
            if 4 == i and val < -0.9:
                ANGLE_4 -= STEP
                arm.set_angle(4, ANGLE_4)
                print("  → 오른쪽 스틱 위로!")
            if 4 == i and val > 0.9:
                ANGLE_4 += STEP
                arm.set_angle(4, ANGLE_4)
                print("  → 오른쪽 스틱 아래로!")
            if 2 == i and val > 0.9:
                ANGLE_2 -= STEP
                arm.set_angle(2, ANGLE_2)
                print("  → 왼쪽 트리거!")
            if 5 == i and val > 0.9:
                ANGLE_2 += STEP
                arm.set_angle(2, ANGLE_2)
                print("  → 오른쪽 트리거!")
            print(f"Axis {i}: {dir_label(i, s)} (value={val:.2f})")
    else:
        print("축: 없음")

    print("=====================\n")
