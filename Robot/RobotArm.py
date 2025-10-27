# pip install pyserial
import serial
import time

class RobotArm:
    """
    직렬 포트로 '모터ID 펄스' 형식("3 1400\\n") 전송하는 경량 드라이버.
    - 각도 입력: set_angle(id, deg, smooth=True, step_deg=3, delay=0.02)
      * smooth=True(기본): 스텝 기반으로 부드럽게 이동
      * 첫 호출(이전 값 없음)은 목표값을 시작점으로 간주 → 점프 없음
    - 펄스 입력: set_pulse(id, us, strict=False)  # strict=True면 범위 밖 예외
    - 안전 범위(us_min~us_max)로 항상 클램프(기본 1000~2000μs)
    - 모터별 offset/역방향 지원
    """
    def __init__(self, port, baudrate=115200, us_min=544, us_max=2400, write_delay=1.0):
        self.ser = serial.Serial(port, baudrate, timeout=0.2)
        self.us_min = int(us_min)
        self.us_max = int(us_max)
        if self.us_min >= self.us_max:
            raise ValueError("us_min < us_max 여야 합니다.")
        self.write_delay = float(write_delay)

        # 모터별 튜닝(선택)
        self.offset_deg = {}   # {motor_id:int -> offset_deg:float}
        self.reversed = {}     # {motor_id:int -> bool}

        # 마지막으로 보낸 펄스 기록
        self._last_us = {}     # {motor_id:int -> us:int}

    # ---------- 내부 유틸 ----------
    def _clamp_us(self, us: int) -> int:
        return max(self.us_min, min(self.us_max, int(us)))

    def _deg_to_us(self, deg: float) -> int:
        # 0~180 범위로 클램프 후 선형 매핑
        d = max(0.0, min(180.0, float(deg)))
        span = self.us_max - self.us_min
        return int(round(self.us_min + span * (d / 180.0)))

    def _us_to_deg(self, us: int) -> float:
        # us→deg 역매핑 (0~180 기준)
        u = float(self._clamp_us(us))
        span = self.us_max - self.us_min
        return 180.0 * (u - self.us_min) / span

    def _apply_offset_rev(self, motor_id: int, deg: float) -> float:
        # 사용자가 준 "논리 각도"에 오프셋/역방향을 반영해 실제 명령 각도 계산
        d = float(deg) + float(self.offset_deg.get(motor_id, 0.0))
        if self.reversed.get(motor_id, False):
            d = 180.0 - d
        # 최종 명령 각도는 0~180으로 제한
        return max(0.0, min(180.0, d))

    def _send(self, motor_id: int, us_value: int):
        self.ser.write(f"{int(motor_id)} {int(us_value)}\n".encode("ascii"))
        if self.write_delay:
            time.sleep(self.write_delay)

    # ---------- 퍼블릭 API ----------
    def set_angle(self, motor_id: int, deg: float, smooth: bool = True, step_deg: float = 3.0, delay: float = 1.0):
        """
        모터ID를 deg(°)로 이동.
        smooth=True일 때 스텝 기반으로 부드럽게 이동.
        - 이전 값이 없으면(초기) 목표값을 시작점으로 간주하여 점프/스윕 생략.
        """
        motor_id = int(motor_id)
        # 오프셋/역방향 반영된 실제 명령 각도
        target_cmd_deg = self._apply_offset_rev(motor_id, deg)
        target_us = self._clamp_us(self._deg_to_us(target_cmd_deg))

        # 시작점 결정: 기록 있으면 그걸, 없으면 이번 목표(=첫 호출은 스윕 없음)
        if motor_id in self._last_us:
            start_us = self._last_us[motor_id]
        else:
            start_us = target_us

        if not smooth:
            # 바로 전송
            self._send(motor_id, target_us)
            self._last_us[motor_id] = target_us
            return

        # 스무스 모션: 시작/목표를 각도 기준으로 만들고 step_deg로 분할
        start_cmd_deg = self._us_to_deg(start_us)  # 이미 offset/rev 반영된 상태로 봐도 일관성 충분
        if abs(target_cmd_deg - start_cmd_deg) <= max(0.5, float(step_deg)):
            # 차이가 작으면 바로 쏨
            self._send(motor_id, target_us)
            self._last_us[motor_id] = target_us
            return

        step = abs(float(step_deg))
        step = 0.1 if step <= 0 else step  # 0 방지
        direction = 1.0 if target_cmd_deg > start_cmd_deg else -1.0

        angle = start_cmd_deg
        # 과도한 반복 방지(최대 단계 제한)
        max_steps = int(abs(target_cmd_deg - start_cmd_deg) / step) + 2
        steps = 0

        while (direction > 0 and angle < target_cmd_deg) or (direction < 0 and angle > target_cmd_deg):
            angle += direction * step
            # 마지막 스텝을 넘어가지 않도록 정리
            if (direction > 0 and angle > target_cmd_deg) or (direction < 0 and angle < target_cmd_deg):
                angle = target_cmd_deg
            us = self._clamp_us(self._deg_to_us(angle))
            self._send(motor_id, us)
            self._last_us[motor_id] = us
            steps += 1
            if steps > max_steps:
                break
            time.sleep(delay)

        # 보증 도착(혹시 루프 탈출한 경우 대비)
        if self._last_us.get(motor_id) != target_us:
            self._send(motor_id, target_us)
            self._last_us[motor_id] = target_us

    def set_angles(self, mapping: dict, smooth: bool = True, step_deg: float = 3.0, delay: float = 1.0):
        """여러 모터 한 번에: {id: deg, ...}"""
        for mid, d in mapping.items():
            self.set_angle(int(mid), float(d), smooth=smooth, step_deg=step_deg, delay=delay)

    def set_pulse(self, motor_id: int, us: int, strict: bool = False):
        """
        펄스를 직접 전송. 안전범위 밖이면:
          - strict=False(기본): 자동 클램프 후 전송
          - strict=True: ValueError 발생
        """
        motor_id = int(motor_id)
        us_int = int(us)
        if strict and not (self.us_min <= us_int <= self.us_max):
            raise ValueError(f"펄스 {us_int}μs 가 안전범위 [{self.us_min}..{self.us_max}] 밖입니다.")
        us_clamped = self._clamp_us(us_int)
        self._send(motor_id, us_clamped)
        self._last_us[motor_id] = us_clamped  # 펄스 직접 보낸 경우에도 상태 갱신

    # 튜닝 옵션
    def set_offset(self, motor_id: int, deg_offset: float):
        self.offset_deg[int(motor_id)] = float(deg_offset)

    def set_reversed(self, motor_id: int, flag: bool = True):
        self.reversed[int(motor_id)] = bool(flag)

    def set_limits(self, us_min: int, us_max: int):
        """안전 펄스 범위 갱신(예: 500-2500으로 확대)"""
        us_min = int(us_min); us_max = int(us_max)
        if us_min >= us_max:
            raise ValueError("us_min < us_max 여야 합니다.")
        self.us_min, self.us_max = us_min, us_max

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

if __name__ == "__main__":
    arm = RobotArm("/dev/ttyUSB0", 9600, us_min=544, us_max=2400)

    # 기본은 smooth=True라서 부드럽게 감
    arm.set_angle(0, 120)                 # 모터 0을 120°
    arm.set_angle(0, 30, step_deg=2)      # 2° 스텝으로 더 부드럽게
    arm.set_angles({1: 45, 2: 10})        # 여러 개도 한 번에(스무스)

    # 초기에는 이전값이 없으니 목표값을 시작점으로 간주 → 스윕 없이 딱 한 번 전송됨
    arm.set_pulse(3, 1400)                # 펄스 직접(안전범위 클램프)
    # arm.set_pulse(3, 2600, strict=True) # → 범위 밖이면 예외

    arm.close()
