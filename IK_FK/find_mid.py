import os

def find_matching_lines(filename='./IK_FK/50_jump_result_same_last_y.txt'):
    """
    파일을 읽어_로 연결된 숫자 그룹이 특정 범위에 속하는 라인을 찾습니다.
    
    조건:
    1. 라인의 첫 번째 단어 (예: 130_105_35)
    2. '⇨' 기호 다음의 첫 번째 단어 (예: 88_177_5)
    3. 이 두 단어에 포함된 _로 구분된 모든 숫자가
    4. 40 이상(>= 40)이고 130 미만(< 130)이어야 합니다.
    """
    
    # 조건 범위 설정
    min_val = 40
    max_val = 130 # 130 '미만'

    # 파일 존재 확인
    if not os.path.exists(filename):
        print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
        # 예시로 가상 파일 생성
        print("예시를 위해 가상 파일을 생성하여 테스트합니다.")
        dummy_data = [
            "130_105_35 (0.0, 92.0),(... ⇨ 88_177_5 (0.0, 92.0),(...", # 조건 미달 (130, 105, 35, 88, 177, 5)
            "50_60_129 (0.0, 92.0),(... ⇨ 70_80_90 (0.0, 92.0),(...", # 조건 충족 (모두 40~129)
            "40_100_90 (0.0, 92.0),(... ⇨ 45_130_88 (0.0, 92.0),(...", # 조건 미달 (130)
            "39_100_90 (0.0, 92.0),(... ⇨ 45_120_88 (0.0, 92.0),(...", # 조건 미달 (39)
            "77_88_99 (0.0, 92.0),(... ⇨ 41_52_63 (0.0, 92.0),(..."  # 조건 충족
        ]
        process_data(dummy_data, min_val, max_val)
        return

    # 파일이 존재할 경우
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"--- '{filename}' 파일에서 조건 검색 결과 ---")
            process_data(lines, min_val, max_val)
            
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")

def process_data(lines, min_val, max_val):
    """주어진 데이터 라인들을 처리하고 조건에 맞는 라인을 출력합니다."""
    
    found_count = 0
    for line in lines:
        line = line.strip()
        if not line or '⇨' not in line:
            continue # 빈 줄이나 화살표가 없는 줄은 건너뜀

        parts = line.split()
        
        # '⇨' 기호의 인덱스를 찾습니다.
        try:
            arrow_index = parts.index('⇨')
        except ValueError:
            continue # '⇨' 기호가 없으면 건너뜀

        # 필요한 토큰(단어)을 추출합니다.
        # parts[0] : 첫 번째 단어 (예: 130_105_35)
        # parts[arrow_index + 1] : 화살표 다음 단어 (예: 88_177_5)
        if arrow_index + 1 >= len(parts):
            continue # 화살표 뒤에 단어가 없으면 건너뜀

        token1 = parts[0]
        token2 = parts[arrow_index + 1]

        # 두 토큰의 모든 숫자들을 리스트로 만듭니다.
        all_numbers_str = token1.split('_') + token2.split('_')
        
        try:
            # 모든 숫자가 조건을 만족하는지 검사
            # all() 함수: 리스트의 모든 요소가 True일 때만 True 반환
            condition_met = all(min_val <= int(num) < max_val for num in all_numbers_str)
            
            if condition_met:
                print(line)
                found_count += 1

        except ValueError:
            # int() 변환 실패 시 (예: '50_abc_60')
            continue # 해당 줄은 무시

    if found_count == 0:
        print("조건에 맞는 행을 찾지 못했습니다.")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    find_matching_lines()