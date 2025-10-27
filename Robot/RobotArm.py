# pip install pyserial
import serial
import time

class RobotArm:
    """
    '모터ID 펄스'("3 1400\\n") 전송하는 경량 드라이버.
    - set_angle(id, deg, smooth=True, step_deg=3.0, delay=0.02)
      * smooth=True(기본): 스텝 기반 부드러운 이동
      * 첫 호출은 이전값 없으므로 목표값을 시작점으로 간주(초기 스윕 없음)
    - set_pulse(id, us, strict=False): 펄스 직접(안전범위 클램프/엄격모드)
    - ★ 매핑 범위(map_us_min/max)와 안전(제한) 범위(safe_us_min/max)를 분리
        예) 매핑: 500–2500 (실제 서보 스펙)
            안전: 544–2400 (Arduino Servo의 안전가드)
    - 모터별 offset/역방향 지원
    """
    def __init__(
        self,
        port,
        baudrate=115200,
        map_us_min=500,      # ← 각도→펄스 '계산'에 쓰는 범위
        map_us_max=2500,
        safe_us_min=544,     # ← 실제 전송 '클램프'에 쓰는 안전범위
        safe_us_max=2400,
        write_delay=0.0
    ):
        self.ser = serial.Serial(port, baudrate, timeout=0.2)

        # 매핑 범위(계산용)
        self.map_us_min = int(map_us_min)
        self.map_us_max = int(map_us_max)
        if self.map_us_min >= self.map_us_max:
            raise ValueError("map_us_min < map_us_max 여야 합니다.")

        # 안전(제한) 범위(클램프용)
        self.safe_us_min = int(safe_us_min)
        self.safe_us_max = int(safe_us_max)
        if self.safe_us_min >= self.safe_us_max:
            raise ValueError("safe_us_min < safe_us_max 여야 합니다.")

        self.write_delay = float(write_delay)

        # 모터별 튜닝(선택)
        self.offset_deg = {}   # {motor_id:int -> offset_deg:float}
        self.reversed = {}     # {motor_id:int -> bool}

        # 마지막으로 보낸 펄스 기록(스무스 시작점 판단용)
        self._last_us = {}     # {motor_id:int -> us:int}

    # ---------- 내부 유틸 ----------
    def _clamp_safe(self, us: int) -> int:
        """안전(제한) 범위로 강제."""
        return max(self.safe_us_min, min(self.safe_us_max, int(us)))

    def _deg_to_us_map(self, deg: float) -> int:
        """각도(0~180) → (매핑)펄스. 계산은 map_us_min~map_us_max 사용."""
        d = max(0.0, min(180.0, float(deg)))
        span = self.map_us_max - self.map_us_min
        return int(round(self.map_us_min + span * (d / 180.0)))

    def _us_to_deg_map(self, us: int) -> float:
        """(매핑)펄스 → 각도(0~180). 역변환도 map 범위를 기준."""
        u = float(max(self.map_us_min, min(self.map_us_max, int(us))))
        span = self.map_us_max - self.map_us_min
        return 180.0 * (u - self.map_us_min) / span

    def _apply_offset_rev(self, motor_id: int, deg: float) -> float:
        d = float(deg) + float(self.offset_deg.get(motor_id, 0.0))
        if self.reversed.get(motor_id, False):
            d = 180.0 - d
        return max(0.0, min(180.0, d))

    def _send(self, motor_id: int, us_value: int):
        self.ser.write(f"{int(motor_id)} {int(us_value)}\n".encode("ascii"))
        if self.write_delay:
            time.sleep(self.write_delay)

    # ---------- 퍼블릭 API ----------
    def set_angle(self, motor_id: int, deg: float, smooth: bool = True, step_deg: float = 3.0, delay: float = 0.02):
        """
        모터ID를 deg(°)로 이동.
        1) 오프셋/역방향 반영 → 2) 매핑범위로 펄스 계산 → 3) 안전범위로 클램프 후 전송.
        smooth=True면 스텝 분할. 첫 호출은 목표를 시작점으로 간주.
        """
        motor_id = int(motor_id)
        target_cmd_deg = self._apply_offset_rev(motor_id, deg)

        # 매핑 범위로 계산
        target_us_mapped = self._deg_to_us_map(target_cmd_deg)
        # 안전 범위로 클램프
        target_us = self._clamp_safe(target_us_mapped)

        # 시작점: 기록 있으면 그 값, 없으면 이번 목표(첫 호출 스윕 없음)
        start_us = self._last_us[motor_id] if motor_id in self._last_us else target_us

        if not smooth:
            self._send(motor_id, target_us)
            self._last_us[motor_id] = target_us
            return

        # 스무스 모션: 시작/목표를 '각도(매핑 기준)'로 비교
        start_cmd_deg = self._us_to_deg_map(start_us)
        diff = abs(target_cmd_deg - start_cmd_deg)

        if diff <= max(0.5, float(step_deg)):
            self._send(motor_id, target_us)
            self._last_us[motor_id] = target_us
            return

        step = abs(float(step_deg)) or 0.1
        direction = 1.0 if target_cmd_deg > start_cmd_deg else -1.0

        angle = start_cmd_deg
        max_steps = int(diff / step) + 2  # 안전장치
        steps = 0

        while (direction > 0 and angle < target_cmd_deg) or (direction < 0 and angle > target_cmd_deg):
            angle += direction * step
            if (direction > 0 and angle > target_cmd_deg) or (direction < 0 and angle < target_cmd_deg):
                angle = target_cmd_deg

            # 각 스텝마다: 매핑 → 안전클램프 → 전송
            us_step = self._deg_to_us_map(angle)
            us_step = self._clamp_safe(us_step)
            self._send(motor_id, us_step)
            self._last_us[motor_id] = us_step

            steps += 1
            if steps > max_steps:
                break
            time.sleep(delay)

        # 보증 도착
        if self._last_us.get(motor_id) != target_us:
            self._send(motor_id, target_us)
            self._last_us[motor_id] = target_us

    def set_angles(self, mapping: dict, smooth: bool = True, step_deg: float = 3.0, delay: float = 0.02):
        for mid, d in mapping.items():
            self.set_angle(int(mid), float(d), smooth=smooth, step_deg=step_deg, delay=delay)

    def set_pulse(self, motor_id: int, us: int, strict: bool = False):
        """
        펄스를 직접 전송.
          - strict=False: 안전범위로 자동 클램프
          - strict=True: 안전범위 밖이면 ValueError
        """
        motor_id = int(motor_id)
        us_int = int(us)
        if strict and not (self.safe_us_min <= us_int <= self.safe_us_max):
            raise ValueError(f"펄스 {us_int}μs 가 안전범위 [{self.safe_us_min}..{self.safe_us_max}] 밖입니다.")
        us_clamped = self._clamp_safe(us_int)
        self._send(motor_id, us_clamped)
        self._last_us[motor_id] = us_clamped

    # ------ 튜닝 메서드 ------
    def set_offset(self, motor_id: int, deg_offset: float):
        self.offset_deg[int(motor_id)] = float(deg_offset)

    def set_reversed(self, motor_id: int, flag: bool = True):
        self.reversed[int(motor_id)] = bool(flag)

    def set_map_limits(self, map_us_min: int, map_us_max: int):
        """각도→펄스 '계산' 범위(실제 서보 스펙에 맞춤)."""
        mmin = int(map_us_min); mmax = int(map_us_max)
        if mmin >= mmax:
            raise ValueError("map_us_min < map_us_max 여야 합니다.")
        self.map_us_min, self.map_us_max = mmin, mmax

    def set_safe_limits(self, safe_us_min: int, safe_us_max: int):
        """전송 '클램프' 범위(보호용 가드)."""
        smin = int(safe_us_min); smax = int(safe_us_max)
        if smin >= smax:
            raise ValueError("safe_us_min < safe_us_max 여야 합니다.")
        self.safe_us_min, self.safe_us_max = smin, smax

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

arm = RobotArm("/dev/ttyUSB0",
               baudrate=9600,
               map_us_min=500, map_us_max=2500,   # 매핑 범위(실서보)
               safe_us_min=544, safe_us_max=2400) # 안전(제한) 범위

arm.set_angle(0, 90)          # 매핑: 500~2500으로 계산 → 전송 전 544~2400으로 클램프
arm.set_angle(0, 180)         # 2500 계산돼도 2400으로 제한되어 나감
arm.set_pulse(1, 2450)        # 직접 펄스 → 2400으로 클램프됨
arm.set_pulse(1, 2450, strict=True)  # → 예외 발생(ValueError)

# 범위 바꾸고 싶으면
arm.set_map_limits(600, 2400)      # 계산범위 변경
arm.set_safe_limits(600, 2400)     # 안전범위 변경
