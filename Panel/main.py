from machine import Pin
import utime as time

# 내장 LED 핀: Pico W면 "LED", 일반 Pico면 25
try:
    led = Pin("LED", Pin.OUT)   # Pico W
except:
    led = Pin(25, Pin.OUT)      # Pico

# 스위치 입력 (내부 풀업)
btn4 = Pin(4, Pin.IN, Pin.PULL_UP)
btn5 = Pin(5, Pin.IN, Pin.PULL_UP)

# 출력 핀 (MOSFET 게이트나 LED)
out6 = Pin(6, Pin.OUT)
out7 = Pin(7, Pin.OUT)

state6 = 0
state7 = 0

# 5초마다 1초 ON (논블로킹)
last_mark = time.ticks_ms()
led_on = False

while True:
    # 버튼 4
    if btn4.value() == 0:
        state6 ^= 1
        out6.value(state6)
        print("Pin6 =", state6)
        time.sleep(0.3)  # 간단 디바운스

    # 버튼 5
    if btn5.value() == 0:
        state7 ^= 1
        out7.value(state7)
        print("Pin7 =", state7)
        time.sleep(0.3)

    # 주기 LED: 5초마다 켜고 1초 뒤 끄기
    now = time.ticks_ms()
    if not led_on and time.ticks_diff(now, last_mark) >= 5000:
        led.value(1)        # 켜기
        led_on = True
        last_mark = now
    elif led_on and time.ticks_diff(now, last_mark) >= 1000:
        led.value(0)        # 끄기
        led_on = False
        last_mark = now
