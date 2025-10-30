# safe_gamepad.py
import pygame, time

pygame.init()
pygame.joystick.init()
pygame.display.set_mode((200, 100))

print("컨트롤러 대기 중... (10초 동안)")
js = None
t_end = time.time() + 10
while time.time() < t_end and js is None:
    # 이벤트 체크
    for e in pygame.event.get():
        if e.type == pygame.JOYDEVICEADDED:
            idx = e.device_index
            js = pygame.joystick.Joystick(idx)
            js.init()
            print(f"[ADDED] idx={idx} name={js.get_name()}")
    # 혹시 이미 연결되어 있으면 바로 잡기
    if pygame.joystick.get_count() > 0 and js is None:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"[ENUM] name={js.get_name()}")
    time.sleep(0.1)

if js is None:
    raise SystemExit("10초 안에 컨트롤러가 안 잡힘")

print("axes:", js.get_numaxes(), "buttons:", js.get_numbuttons(), "hats:", js.get_numhats())

# 이벤트 루프
try:
    while True:
        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                print("BTN↓", e.button)
            elif e.type == pygame.JOYBUTTONUP:
                print("BTN↑", e.button)
            elif e.type == pygame.JOYAXISMOTION:
                print("AXIS", e.axis, round(e.value, 3))
            elif e.type == pygame.JOYHATMOTION:
                print("HAT", e.hat, e.value)
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
