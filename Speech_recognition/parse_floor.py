# parse_floor.py
import re

SINO = {"공":0,"영":0,"일":1,"이":2,"삼":3,"사":4,"오":5,"육":6,"륙":6,"칠":7,"팔":8,"구":9}
NATIVE = {
    "한":1,"두":2,"세":3,"네":4,"다섯":5,"여섯":6,"일곱":7,"일럭":7,"여덟":8,"아홉":9,"열":10,
    "열한":11,"열두":12,"열세":13,"열네":14,"열다섯":15,"열여섯":16,"열일곱":17,"열여덟":18,"열아홉":19
}

def parse_korean_number_word(word: str) -> int | None:
    word = word.replace(" ", "")
    # 1) Native 우선 체크 (1~19 자주 쓰는 층 범위)
    for k, v in sorted(NATIVE.items(), key=lambda x: -len(x[0])):
        if word.startswith(k):
            return v
    # 2) Sino-Korean (최대 99 처리: (X)십(Y))
    # 예: "십오"=15, "이십삼"=23, "십"=10
    if "십" in word:
        left, _, right = word.partition("십")
        tens = SINO.get(left, 1) if left != "" else 1
        ones = SINO.get(right, 0) if right != "" else 0
        return tens * 10 + ones
    # 일~구 단독
    if word in SINO:
        return SINO[word]
    return None

def parse_floor(text: str) -> int | None:
    t = text.strip()

    # 1) 숫자 직접 표기: "15층", "15 층"
    m = re.search(r'(\d+)\s*층', t)
    if m:
        return int(m.group(1))

    # 2) 한글 표기: "...층" 앞부분만 떼서 숫자 해석 시도
    m2 = re.search(r'([가-힣]+)\s*층', t)
    if m2:
        w = m2.group(1)
        # 길이가 길면 뒤에서부터 잘라보며 탐색 (예: "이십삼" 포함 문맥)
        for L in range(min(3, len(w)), 0, -1):   # 3글자까지 우선 탐색
            n = parse_korean_number_word(w[-L:])
            if n is not None:
                return n

        # 그래도 못 찾았으면 전체 시도
        n = parse_korean_number_word(w)
        if n is not None:
            return n

    return None

if __name__ == "__main__":
    tests = ["5층 눌러줘", "십오층 부탁", "열두 층", "한 층", "이십삼층 가자", "B1층", "삼 층 눌러"]
    for s in tests:
        print(s, "=>", parse_floor(s))
