import re
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------
## 📈 메인 시각화 함수 정의
# -----------------------------------------------------------------
def plot_points(file_path, line_start=None, line_end=None, target_points=None):
    """
    파일에서 좌표를 읽어 점을 플롯하고, 지정된 경우 선과 다중 타겟 점을 그립니다.

    :param file_path: 좌표 데이터가 포함된 파일 경로
    :param line_start: 그릴 선의 시작점 (x1, y1) 튜플, 기본값: None
    :param line_end: 그릴 선의 끝점 (x2, y2) 튜플, 기본값: None
    :param target_points: 강조하여 그릴 단일 또는 다중 점 [(x, y), (x, y), ...], 기본값: None
    """
    points = []

    try:
        # 1. 파일에서 좌표 데이터 읽기 및 추출
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 마지막 괄호 안 좌표 추출
                match = re.findall(r"\(([^)]+)\)", line)
                if match:
                    last = match[-1]
                    
                    try:
                        x, y = map(float, last.split(","))
                        points.append((x, y))
                    except ValueError:
                        print(f"[경고] 잘못된 좌표 형식 무시: {last}")
                        continue
                        
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
        return

    if not points:
        print("[정보] 파일에서 유효한 좌표를 찾지 못했습니다.")
        return

    # 2. x, y 좌표 분리
    xs, ys = zip(*points)

    # 3. 플롯 설정 (디자인 변경 반영)
    plt.style.use('seaborn-v0_8-whitegrid')  # 깔끔한 그리드 스타일 적용
    plt.figure(figsize=(12, 8))
    
    # 3-1. 기존 점 플롯 (색상 및 투명도 변경)
    plt.scatter(xs, ys, color="#1f77b4", marker=".", alpha=0.6, label="Extracted Points")
    
    # 3-2. 선택적 선 그리기 (기존 로직 유지)
    if line_start and line_end:
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 선 그리기: 어두운 색상 사용
        plt.plot([x1, x2], [y1, y2], color="#d62728", linestyle="-", linewidth=2.5, label="Reference Line", zorder=4)
        
    # 3-3. 다중 타겟 점 플롯 및 인덱스 표시 (수정/추가된 부분)
    if target_points:
        # 리스트가 아닌 단일 튜플이 들어온 경우 리스트로 변환하여 처리 가능하도록 함
        if not isinstance(target_points, list):
             target_points = [target_points]
             
        txs = [p[0] for p in target_points]
        tys = [p[1] for p in target_points]
        
        # 타겟 점 플롯 (큰 원형 마커 사용)
        plt.scatter(txs, tys, color="#ff7f0e", marker="o", s=150, edgecolors='black', label="Target Points", zorder=5) 
        
        print(f"[정보] 다중 타겟 점 플롯: {target_points}")
        
        # 인덱스 번호 표시 (annotate)
        for i, (px, py) in enumerate(target_points):
            # 점 위에 인덱스 번호 텍스트 추가 (0.015만큼 상단에 표시)
            plt.annotate(
                f'{i}', 
                (px, py), 
                textcoords="offset points", 
                xytext=(10, 0), 
                ha='center', 
                fontsize=12, 
                fontweight='bold',
                color="#ff7f0e"
            )

    # 4. 그래프 레이블 및 제목
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)
    plt.title("Robot Arm Workspace Visualization", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    # xlim, ylim을 설정하여 여백을 확보
    all_x = list(xs) + txs if target_points else list(xs)
    all_y = list(ys) + tys if target_points else list(ys)
    
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 10% 여백 추가
        plt.xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
        plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    plt.legend()
    
    # 5. 그래프 표시
    plt.show()

# -----------------------------------------------------------------
## ⚙️ 실행 부분 (main)
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    # ⚠️ 파일 경로 설정
    FILE_PATH = "IK_FK/angles_coords_step1.txt"
    
    print(f"파일 경로: {os.path.abspath(FILE_PATH)}")

    # 여러 개의 타겟 점 정의 (리스트로 전달)
    TARGET_POINTS_LIST = [
        (-140 + 85, 300 + 55), #cam 1
        (-140 + 85, 270 + 55), #cam 2  
        (-246, 78 + 254), #button의 x좌표는 연산된 결과, y좌표는 두유ㅋㅋ 오프셋 + a4용지상 버튼 위치 (78 + 254)
    ]
    
    START_POINT = (-246, 0)
    END_POINT = (-246, 380)
    
    print(f"\n--- 1. 다중 타겟 점 및 선 포함 플롯 실행 ---")
    plot_points(
        FILE_PATH, 
        line_start=START_POINT, 
        line_end=END_POINT, 
        target_points=TARGET_POINTS_LIST 
    )
