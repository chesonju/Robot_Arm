# -*- coding: utf-8 -*-
import glob, os, cv2, numpy as np

# ===================================================================
# 1. 체커보드 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (9, 6)

# 2. 체커보드 한 칸 크기 (mm 단위)
square_size = 24.0

# 3. 시각화 결과 저장할 폴더
output_dir = 'calibration_visualization'
os.makedirs(output_dir, exist_ok=True)
# ===================================================================

# 다양한 확장자 포함
patterns = [
    './Camera/chessboard_photos/*.jpg',
    './Camera/chessboard_photos/*.JPG',
    './Camera/chessboard_photos/*.jpeg',
    './Camera/chessboard_photos/*.JPEG',
    './Camera/chessboard_photos/*.png',
    './Camera/chessboard_photos/*.PNG',
]
images = []
for p in patterns:
    images.extend(glob.glob(p))
images = sorted(images)

print(f"총 {len(images)}개의 이미지를 불러왔습니다. 캘리브레이션을 시작합니다...")

# 체커보드 3D 포인트 준비
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints = [], []
save_ok = 0
success_cnt = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"❌ 로드 실패: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        success_cnt += 1
        print(f"✅ {fname} 에서 코너 검출 성공!")

        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)

        base = os.path.basename(fname)
        stem, ext = os.path.splitext(base)
        out_name = f"{stem}_calibrated{ext.lower()}"
        output_path = os.path.join(output_dir, out_name)

        if cv2.imwrite(output_path, vis):
            save_ok += 1
        else:
            print(f"❌ 저장 실패: {output_path}")
    else:
        print(f"❌ {fname} 에서 코너 검출 실패...")

print(f"\n[요약] 코너 검출 성공: {success_cnt}장, 시각화 저장 성공: {save_ok}장")

# ──────────────────────────────────────────────────────────────
#  캘리브레이션 실행
# ──────────────────────────────────────────────────────────────
if objpoints and imgpoints:
    print("\n카메라 캘리브레이션 실행 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n========== 카메라 캘리브레이션 결과 ==========")
    print("## 1. 카메라 행렬 (Camera Matrix)")
    print(mtx)
    print("\n## 2. 왜곡 계수 (Distortion Coefficients)")
    print(dist)

    fx, fy, cx, cy = mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]
    print("\n## 3. 주요 파라미터 요약")
    print(f">> 초점 거리 (Focal Length): fx = {fx:.4f}, fy = {fy:.4f}")
    print(f">> 주점 (Principal Point): cx = {cx:.4f}, cy = {cy:.4f}")

    # ─────────────────────────────────────────────
    # 🔎 리프로젝션 에러 계산
    # ─────────────────────────────────────────────
    per_view_errors = []
    total_err = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2) / len(imgpts2)
        per_view_errors.append((images[i], float(err)))
        total_err += err * len(imgpts2)
        total_points += len(imgpts2)

    mean_err = total_err / total_points
    print("\n======== 리프로젝션 에러 ========")
    print(f"OpenCV RMS (ret): {ret:.6f} px")
    print(f"Mean L2 per-corner: {mean_err:.6f} px")

    per_view_errors.sort(key=lambda x: x[1], reverse=True)
    worst_k = min(10, len(per_view_errors))
    print(f"\n오차 큰 상위 {worst_k}장:")
    for path, e in per_view_errors[:worst_k]:
        print(f"  {os.path.basename(path):40s}  {e:.6f} px")

    # 텍스트 저장
    with open('./Camera/reprojection_error_report.txt', 'w', encoding='utf-8') as f:
        f.write("======== Reprojection Error Report ========\n")
        f.write(f"OpenCV RMS (ret): {ret:.6f} px\n")
        f.write(f"Mean L2 per-corner: {mean_err:.6f} px\n\n")
        f.write("Per-image mean L2 error (px):\n")
        for path, e in per_view_errors:
            f.write(f"{os.path.basename(path)}, {e:.6f}\n")

    print("\n📝 리프로젝션 에러 리포트: Camera/reprojection_error_report.txt 저장 완료")
else:
    print("⚠️ 유효한 코너를 찾지 못했습니다.")
