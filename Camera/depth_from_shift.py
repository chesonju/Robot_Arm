# file: depth_from_shift.py
from dataclasses import dataclass
import numpy as np
import cv2
from typing import Optional, Tuple, Tuple, Union, Sequence

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[np.ndarray] = None

# ✅ 캘리브레이션 결과
K = np.array([
    [304.92065479,   0.0,           156.23274445],
    [  0.0,         313.68343277,   126.36038472],
    [  0.0,           0.0,             1.0      ]
], dtype=np.float64)

dist = np.array([[-9.67711486e-03, -2.24344024e-01, -1.53467994e-03,
                   6.54834768e-04,  1.48537539e+00]], dtype=np.float64)

intr = CameraIntrinsics(
    fx=K[0, 0],
    fy=K[1, 1],
    cx=K[0, 2],
    cy=K[1, 2],
    dist=dist
)

def undistort_points(pts: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    if intr.dist is None:
        return pts.copy()
    K = np.array([[intr.fx, 0, intr.cx],
                  [0, intr.fy, intr.cy],
                  [0,       0,      1]], dtype=np.float64)
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, intr.dist, P=K)
    return undist.reshape(-1, 2)


def _get_y(center: Union[Tuple[float, float], Sequence[Tuple[float, float]], Sequence[float]]) -> float:
    """
    center가 (x, y) 또는 [(x, y), ...] 둘 다 대응해서 y를 반환.
    """
    # (x, y) 형태
    if isinstance(center, (list, tuple)) and len(center) >= 2 and isinstance(center[0], (int, float)):
        return float(center[1])
    # [(x, y), ...] 형태
    if isinstance(center, (list, tuple)) and center and isinstance(center[0], (list, tuple)) and len(center[0]) >= 2:
        return float(center[0][1])
    raise TypeError(f"center 형식이 이상함: {center!r}")

def depth_from_vertical_shift(
    delta_y_mm: float,
    first_center: Tuple[float, float],
    second_center: Tuple[float, float],
    undistort: bool = True
) -> float:
    """
    카메라가 delta_y_mm(mm 단위) 만큼 위/아래로 평행이동했을 때,
    같은 물체의 중심 좌표 first_center → second_center 로 바뀌었다면 깊이(Z)를 추정.
    결과 단위는 mm.
    """
    # 1) 입력 정규화: y만 뽑기
    v1 = _get_y(first_center)
    v2 = _get_y(second_center)

    # 2) 필요 시 왜곡 보정
    if undistort and getattr(intr, "dist", None) is not None:
        pts = np.array([[intr.cx, v1], [intr.cx, v2]], dtype=np.float64)  # (N,2)
        und = undistort_points(pts, intr)  # (N,2) 가정: x,y
        v1, v2 = float(und[0, 1]), float(und[1, 1])

    # 3) 픽셀 이동량
    dv = v2 - v1
    if abs(dv) < 1e-9:
        raise ValueError("Δv≈0 → 깊이 계산 불가")

    # 4) 깊이 계산: Z = f_y * ΔY / |Δv|
    # 부호까지 보존하려면 abs 제거하고 Z = (intr.fy * delta_y_mm) / dv 사용
    Z_mm = (intr.fy * delta_y_mm) / abs(dv)
    return float(Z_mm)

# ---------------- 사용 예시 ----------------
if __name__ == "__main__":
    first_center = (200, 130)   # (x, y)
    second_center = (210, 110)  # (x, y)

    Z_mm = depth_from_vertical_shift(50, first_center, second_center)
    print(f"추정 깊이 Z ≈ {Z_mm:.2f} mm")
