import serial # pip install pyserial
import serial.tools.list_ports
import time

# 전역 시리얼 객체
ser = None

def find_and_start_port(baudrate=9600, timeout=1):
    """
    사용 가능한 시리얼 포트를 검색하고, 
    키워드(예: 'USB-SERIAL')와 일치하는 포트에 자동으로 연결을 시도합니다.
    """
    global ser
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("오류: 연결된 시리얼 포트가 없습니다.")
        return False

    print("--- 사용 가능한 시리얼 포트 ---")
    for i, p in enumerate(ports):
        print(f"[{i+1}] {p.device} - {p.description}")
    print("---------------------------------")

    target_port_name = "/dev/cu.usbmodem11101"  # <--- 본인의 포트 이름으로 수정 (예: /dev/ttyUSB0)
    print(f"지정된 포트({target_port_name})에 연결을 시도합니다.")
    try:
        ser = serial.Serial(port=target_port_name, baudrate=baudrate, timeout=timeout)
        time.sleep(2) # 장치 초기화를 위해 잠시 대기
        print(f"시리얼 포트 {ser.port} 연결 성공: {ser.is_open}")
        return True
    except Exception as e:
        print(f"포트({target_port_name}) 연결 실패: {e}")
        return False

    # 위 1번(자동 선택) 로직을 사용하여 포트 열기 시도
    try:
        ser = serial.Serial(
            port=target_port.device,
            baudrate=baudrate,
            timeout=timeout
        )
        # 아두이노 등 일부 장치는 시리얼 연결 시 재부팅되므로 잠시 대기
        time.sleep(2) 
        print(f"시리얼 포트 {ser.port} 연결 성공: {ser.is_open}")
        return True
    except Exception as e:
        print(f"포트({target_port.device}) 연결 실패: {e}")
        return False

def close_port():
    """시리얼 포트가 열려있으면 닫습니다."""
    global ser
    if ser and ser.is_open:
        ser.close()
        print(f"\n시리얼 포트 {ser.port} 연결 해제.")

def angle_to_pulse(angle):
    """각도(0~180)를 펄스 폭(600~2400)으로 변환합니다."""
    # 600 ~ 2400 us
    # 0 deg to 600, 180 deg to 2400
    
    # 각도를 0~180 범위로 강제 조정
    while angle > 180 or angle < 0:
        angle += 180 * (-1 if angle > 180 else 1)

    # 0 -> 600
    # 180 -> 600 + 1800 = 2400
    pulse_width = 600 + (angle * 10)
    
    # 펄스 폭은 보통 정수입니다.
    return int(pulse_width)

def send_angle(servo, angle, speed=0):
    """서보 명령을 포맷하여 시리얼로 전송합니다."""
    global ser
    if not ser or not ser.is_open:
        print("오류: 시리얼 포트가 열려있지 않습니다.")
        return

    pulse = angle_to_pulse(angle)
    
    # --- [중요] 코드 수정 지점 ---
    # 기존 `ser.write(b'%d %f %f', ...)` 방식은 잘못된 문법입니다.
    # 1. f-string으로 보낼 명령 문자열을 만듭니다. (장치에 따라 끝에 '\n'이 필요할 수 있습니다.)
    # 2. `encode('ascii')`를 사용해 문자열을 바이트로 변환합니다.
    command_string = f"{servo} {pulse} {speed}\n"
    
    try:
        ser.write(command_string.encode('ascii'))
        # .strip()은 print 시 \n (줄바꿈)을 제거하기 위함
        print(f"전송: {command_string.strip()}")
    except Exception as e:
        print(f"시리얼 쓰기 오류: {e}")

if __name__ == "__main__":
    # 스크립트 시작 시 자동으로 포트 검색 및 연결
    if find_and_start_port():
        print("\n서보 ID, 각도(0~180), 속도(0~1)를 입력하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")
        
        while True:
            try:
                servo = int(input('Servo ID: '))
                angle = float(input('Angle (0~180): '))
                speed = float(input('Speed (0~1): '))
                
                send_angle(servo, angle, speed)
                
            except KeyboardInterrupt:
                # Ctrl+C 입력 시 루프 종료
                print("\n사용자 요청으로 종료합니다.")
                break
            except ValueError:
                print("잘못된 입력입니다. 숫자를 입력하세요.")
            except Exception as e:
                print(f"알 수 없는 오류 발생: {e}")
        
        # 프로그램 종료 전 포트 닫기
        close_port()
    else:
        print("시리얼 통신을 시작할 수 없습니다. 프로그램을 종료합니다.")