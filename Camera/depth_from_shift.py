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

# ✅ 최신 캘리브레이션 결과 적용
K = np.array([
    [2.22563840e+03, 0.00000000e+00, 9.16584353e+02],
    [0.00000000e+00, 2.23904691e+03, 5.13164209e+02],
    [0.00000000e+0, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float64)

dist = np.array([[ 0.09887097, 0.13530402, -0.00369226, 0.00651958, -1.75474597]],
                dtype=np.float64)

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
    z_est = (intr.fy * delta_y_mm) / abs(dv)

    return float(z_est * 1.2)

import numpy as np
import cv2
import math

def pixel_to_angles_with_undistort(center, K=K, dist=dist):
    """
    K: 3x3 camera matrix, dist: (k1,k2,p1,p2,k3)
    return:
      yaw_deg, pitch_deg, ray (카메라 좌표계에서 z=1로 정규화된 방향벡터)
    """

    u, v = center
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)    # (1,1,2)
    # undistort → 정규화 좌표 (x', y'), z=1의 이미지 평면으로 보면 됨
    und = cv2.undistortPoints(pts, K, dist)                       # (1,1,2)
    x_n, y_n = und[0,0,0], und[0,0,1]

    # 카메라 좌표계의 광선 방향 벡터 (스케일 자유)
    ray = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)

    # 각도 계산 (광학축 z에 대한 각)
    yaw = math.atan2(x_n, 1.0)      # 좌우
    pitch = math.atan2(y_n, 1.0)    # 상하

    return math.degrees(yaw), math.degrees(pitch), ray


# ---------------- 사용 예시 ----------------
if __name__ == "__main__":
    first_center = (978, 197)   # (x, y)
    second_center = (976, 829)  # (x, y)

    z_est = depth_from_vertical_shift(50, first_center, second_center)

    SCALE_Z = 250.0 / 177.49  # ≈ 1.4086
    z_corr = z_est * SCALE_Z

    print(f"추정 깊이 Z ≈ {z_corr:.2f} mm")

    yaw_deg, pitch_deg, ray = pixel_to_angles_with_undistort(second_center)
    print(f"[ANGLE UD] yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°, ray={ray}")