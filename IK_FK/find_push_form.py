import find_target_in_txt
import enhanced_plot
import os, re

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"

def filter_by_angle_conditions(hit_list):
    """
    't1_t2_t3 ...' 형식의 문자열 리스트에서 
    t1 < 90 이고 t2 >= 90 인 경우만 필터링합니다.

    :param hit_list: 필터링할 문자열 리스트
    :return: 조건을 만족하는 문자열의 새 리스트
    """
    matching_hits = []
    
    for hit_string in hit_list:
        try:
            # 1. 공백을 기준으로 첫 번째 부분('t1_t2_t3')을 분리합니다.
            angle_part = hit_string.split(' ')[0]  # 예: '21_180_110'
            
            # 2. '_'를 기준으로 t1, t2, t3를 분리합니다.
            parts = angle_part.split('_')  # 예: ['21', '180', '110']
            
            # 3. t1과 t2를 정수(int)로 변환합니다.
            # (적어도 2개의 부분이 있는지 확인)
            if len(parts) >= 2:
                t1 = int(parts[0])
                t2 = int(parts[1])
                t3 = int(parts[2])
                
                # 4. 조건을 확인합니다.
                if t3 < 90:
                    matching_hits.append(hit_string)
                    
        except (ValueError, IndexError) as e:
            # 숫자로 변환할 수 없거나 형식이 맞지 않는 경우
            print(f"경고: '{hit_string}' 라인을 처리 중 오류 발생 (무시함): {e}")
            
    return matching_hits

def filter_by_geometry(hit_list):
    """
    좌표 리스트에서 3번째 점(P3)이 2번째 점(P2)과 4번째 점(P4)을 잇는 
    직선보다 위에 있는지(Y값이 큰지) 확인하여 필터링합니다.

    :param hit_list: 필터링할 문자열 리스트
    :return: 조건을 만족하는 문자열의 새 리스트
    """
    matching_hits = []
    
    # (x, y) 좌표를 추출하기 위한 정규표현식 (미리 컴파일)
    # 예: (0.0, 94.0)
    coord_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)')
    
    for hit_string in hit_list:
        try:
            # 1. 문자열에서 모든 (x, y) 좌표를 찾습니다.
            #    결과 예: [('0.0', '94.0'), ('-98.0', '131.6'), ('-235.3', '184.3'), ('-374.9', '305.7')]
            coords = coord_pattern.findall(hit_string)
            
            # 2. 2, 3, 4번째 점이 모두 있는지 확인합니다. (인덱스는 1, 2, 3)
            if len(coords) < 4:
                continue # 점이 4개 미만이면 건너뜁니다.

            # 3. P2, P3, P4의 (x, y) 값을 float으로 변환합니다.
            p2_x, p2_y = float(coords[1][0]), float(coords[1][1])
            p3_x, p3_y = float(coords[2][0]), float(coords[2][1])
            p4_x, p4_y = float(coords[3][0]), float(coords[3][1])

            # 4. P2와 P4를 잇는 직선의 기울기(m)를 계산합니다.
            
            # 4a. 수직선인 경우 (ZeroDivisionError 방지)
            if abs(p4_x - p2_x) < 1e-9: 
                # 선이 수직일 때 "위"라는 개념이 모호하므로,
                # 이 경우 P3가 P2-P4의 왼쪽에 있는지(x가 작은지) 등으로 
                # 판단할 수 있으나, 여기서는 우선 건너뜁니다.
                # 필요시: if p3_x < p2_x: matching_hits.append(hit_string)
                continue
                
            # 4b. 일반적인 경우 (기울기 계산)
            m = (p4_y - p2_y) / (p4_x - p2_x)
            
            # 5. P3의 x좌표(p3_x)일 때, P2-P4 직선 위의 y좌표(y_on_line)를 계산합니다.
            #    (점-기울기 공식: y - y1 = m(x - x1)  =>  y = m(x - x1) + y1)
            y_on_line = m * (p3_x - p2_x) + p2_y
            
            # 6. P3의 실제 y좌표(p3_y)가 선 위의 y좌표(y_on_line)보다 큰지 비교합니다.
            if p3_y > y_on_line:
                matching_hits.append(hit_string)
                
        except (ValueError, IndexError) as e:
            # 숫자 변환 실패 또는 좌표가 부족한 경우
            print(f"경고: '{hit_string}' 라인 처리 중 오류 발생 (무시함): {e}")
            
    return matching_hits

def find_and_run_target(target_x, target_y):
    """
    지정된 target_x, target_y에 대해 모든 필터링을 수행하고
    최종 선정된 값으로 GUI를 실행합니다.
    """
    filename = HERE + "angles_coords_step1.txt"
    best_hit = None # 최종 결과를 저장할 변수

    print(f"--- 타겟 검색 시작 (X: {target_x}, Y: {target_y}) ---")

    # 1. 기존 검색 실행
    hits = find_target_in_txt.search_last_coord_in_file(filename, target_x, target_y, tol=1.0)
    print(f"1. 좌표 검색 완료: {len(hits)}개 발견")

    # 2. Y축 임계값으로 필터링
    filtered_hits = find_target_in_txt.filter_by_y_threshold(hits, threshold=30)
    print(f"2. Y축 필터링 완료: {len(filtered_hits)}개 남음")

    # 3. 새로 추가한 각도 조건으로 필터링
    # (이 함수가 이 스크립트 내에 정의되어 있다고 가정)
    angle_filtered_hits = filter_by_angle_conditions(filtered_hits)
    print(f"3. 각도 필터링 완료: {len(angle_filtered_hits)}개 남음")

    # 4. ★새로운 기하학적 조건으로 필터링★ (P3가 P2-P4 선보다 위에 있는 경우)
    # (이 함수가 이 스크립트 내에 정의되어 있다고 가정)
    geometric_filtered_hits = filter_by_geometry(angle_filtered_hits)
    print(f"4. 기하학 필터링 완료: {len(geometric_filtered_hits)}개 남음")

    # 5. 최종 결과 리스트에서 "가운데" 항목 선택
    if geometric_filtered_hits:
        print(f"--- 기하학적 필터링 통과 (총 {len(geometric_filtered_hits)}개) ---")
        
        try:
            # 리스트의 가운데 인덱스 항목을 선택
            middle_index = len(geometric_filtered_hits) // 2
            best_hit = geometric_filtered_hits[middle_index]
            
            print(f"\n--- 🏆 최종 확정 (가운데 {middle_index}번 인덱스 항목) ---")
            print(best_hit)
            
        except (ValueError, IndexError) as e:
            print(f"\n오류: 최종 항목을 선택하는 중 오류가 발생했습니다: {e}")
            return # 함수 종료

    else:
        print("모든 조건(좌표, Y축, 각도, 기하학적 위치)을 만족하는 결과를 찾을 수 없습니다.")
        return # 함수 종료

    # 6. GUI 실행
    if best_hit:
        try:
            # 1번 단계에서 .py 파일에 추가한 parse_thetas 함수 사용
            t1, t2, t3 = find_target_in_txt.parse_thetas(best_hit)
            print(f"\n--- GUI 실행 (t1={t1}, t2={t2}, t3={t3}) ---")
            enhanced_plot.run_arm_gui(t1, t2, t3, block=True)
            print("--- GUI 종료 ---")
            
        except Exception as e:
            print(f"\n오류: Txt 파싱 또는 GUI 실행 중 오류 발생: {e}")
    else:
        print("최종 확정된 'best_hit'이 없어 GUI를 실행할 수 없습니다.")


# =================================================
# 메인 실행 블록
# =================================================
if __name__ == "__main__":
    
    # 여기서 원하는 좌표를 입력하여 함수를 호출합니다.
    result = find_and_run_target(target_x=-243, target_y=347)
    print(result)
    
    # find_target_in_txt.find_target_in_file_to_png(-243, 347)
    
    # 다른 좌표로 또 실행할 수 있습니다.
    # print("\n" + "="*30 + "\n")
    # find_and_run_target(target_x=-300, target_y=300)