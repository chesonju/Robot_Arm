import numpy as np
import cv2
import glob
import os # 폴더 생성을 위해 추가

# ===================================================================
# 이 부분의 값을 사용자의 체커보드에 맞게 수정하세요.
# ===================================================================
# 1. 체커보드의 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (9, 6) # 예: 가로 9개, 세로 6개

# 2. 체커보드 사각형 하나의 실제 크기 (mm 단위)
square_size = 25.0 # 예: 25mm

# 3. 시각화 결과물 및 텍스트 파일을 저장할 폴더 이름
output_dir = 'calibration_visualization'
# ===================================================================

# 출력 폴더가 없다면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3D 공간의 체커보드 코너 좌표를 저장할 배열
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * square_size

# 모든 이미지에서 찾은 3D 'object points'와 2D 'image points'를 저장할 리스트
objpoints = [] # 3D 포인트
imgpoints = [] # 2D 이미지 속 포인트

# 현재 폴더에 있는 모든 jpg 이미지를 불러옵니다.
images = glob.glob('./Camera/chessboard_photos/*.jpg')

if not images:
    print("오류: 현재 폴더에 .jpg 이미지가 없습니다. 체커보드 사진을 폴더에 넣어주세요.")
else:
    print(f"총 {len(images)}개의 이미지를 불러왔습니다. 캘리브레이션을 시작합니다...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        # 코너를 찾았다면
        if ret == True:
            print(f"✅ {fname} 에서 코너 검출 성공!")
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            
            # === 시각화 이미지를 그려서 파일로 저장하는 부분 ===
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # 저장할 파일 경로 생성 (예: calibration_visualization/image1_calibrated.jpg)
            output_path = os.path.join(output_dir, os.path.basename(fname).replace('.jpg', '_calibrated.jpg'))
            cv2.imwrite(output_path, img)
            
        else:
            print(f"❌ {fname} 에서 코너 검출 실패...")

    # 캘리브레이션 실행
    if objpoints and imgpoints:
        print("\n카메라 캘리브레이션을 실행합니다...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("✅ 캘리브레이션 성공! 결과를 'calibration_results.txt' 파일로 저장합니다.")
            
            # === 결과값을 의미와 함께 .txt 파일로 저장하는 부분 ===
            with open('./Camera/calibration_results.txt', 'w', encoding='utf-8') as f:
                f.write("========== 카메라 캘리브레이션 결과 ==========\n\n")
                
                f.write("## 1. 카메라 행렬 (Camera Matrix)\n")
                f.write(" - 카메라의 내부 파라미터를 담고 있는 3x3 행렬입니다.\n")
                f.write(" - fx, fy: 픽셀 단위의 초점 거리\n")
                f.write(" - cx, cy: 이미지의 중심점 (주점)\n")
                f.write(str(mtx) + "\n\n")

                f.write("## 2. 왜곡 계수 (Distortion Coefficients)\n")
                f.write(" - 렌즈로 인해 발생하는 이미지의 왜곡(방사 왜곡, 접선 왜곡)을 설명하는 값들입니다.\n")
                f.write(str(dist) + "\n\n")

                fx = mtx[0, 0]
                fy = mtx[1, 1]
                cx = mtx[0, 2]
                cy = mtx[1, 2]

                f.write("## 3. 주요 파라미터 요약\n")
                f.write(f">> 초점 거리 (Focal Length): fx = {fx:.4f} pixels, fy = {fy:.4f} pixels\n")
                f.write(f">> 주점 (Principal Point): cx = {cx:.4f} pixels, cy = {cy:.4f} pixels\n")
            
            print(f">> 초점 거리 (Focal Length): fx = {fx:.4f} pixels, fy = {fy:.4f} pixels")
            print(f">> 주점 (Principal Point): cx = {cx:.4f} pixels, cy = {cy:.4f} pixels")
        else:
            print("캘리브레이션에 실패했습니다.")

    else:
        print("\n오류: 유효한 체커보드 코너를 하나도 찾지 못했습니다. 사진을 확인해주세요.")

'''
fx, fy: 이것이 바로 우리가 찾던 픽셀 단위의 초점 거리입니다. fx는 x축 초점 거리, 
fy는 y축 초점 거리이며 보통 두 값은 매우 유사합니다. 거리 계산 공식에는 이 fx 또는 fy 값을 사용하면 됩니다.

cx, cy: 이미지의 중심점(주점)의 픽셀 좌표입니다. 일반적으로 이미지 해상도의 절반에 가까운 값이 나옵니다.
'''