import re
import os
import cv2

def parse_find_buttons_results(data_list):
    """
    OCR 결과 문자열 리스트를 파싱해서 dict 리스트로 반환한다.
    포맷: '[score] (x1, y1, x2, y2) -> text (conf)'
    """
    pattern = re.compile(
        r"\[(?P<score>[\d.]+)\]\s*"                          # [score]
        r"\((?P<x1>\d+),\s*(?P<y1>\d+),\s*(?P<x2>\d+),\s*(?P<y2>\d+)\)\s*"  # bbox
        r"->\s*(?P<text>.+?)\s*"                             # text
        r"\((?P<conf>[\d.]+)\)"                              # (conf)
    )

    parsed = []
    for d in data_list:
        m = pattern.search(d)
        if m:
            parsed.append({
                "score": float(m.group("score")),
                "bbox": (
                    int(m.group("x1")),
                    int(m.group("y1")),
                    int(m.group("x2")),
                    int(m.group("y2")),
                ),
                "text": m.group("text"),
                "conf": float(m.group("conf")),
            })
    
    return parsed

def get_bbox_by_text(parsed_results, target_text):
    """
    주어진 target_text와 같은 OCR 결과 중 score가 가장 높은 bbox 반환.
    없으면 None 반환.
    """
    candidates = [r for r in parsed_results if r["text"] == target_text]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["score"])
    return best["bbox"]

def draw_bbox_and_center(image_path, bbox, backup_dir="Image_backup", out_name="step1.png",
                         color=(0, 255, 0), thickness=None, dot_radius=10, save=True):
    """
    bbox: (x1,y1,x2,y2)
    단일 bbox에 초록색 박스와 중심점(굵은 원) 그려서 저장.
    중심 좌표 (cx,cy) 반환.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    # 선 두께 자동 조절
    if thickness is None:
        h, w = img.shape[:2]
        thickness = max(2, min(h, w) // 200)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.circle(img, (cx, cy), dot_radius, (0, 255, 0), -1)

    os.makedirs(backup_dir, exist_ok=True)
    out_path = os.path.join(backup_dir, out_name)
    if save:
        cv2.imwrite(out_path, img)
        print(f'saved: {out_path}\n')

    return (cx, cy)

def print_color(text: str, color: str = "green", bold: bool = True) -> None:
    """
    색깔 있는 문자열 출력
    color: 'red', 'green', 'yellow', 'blue', 'magenta', 'pink', 'cyan', 'white'
    bold: True면 굵게 출력
    """
    colors = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "pink": 35,      # pink = magenta
        "cyan": 36,
        "white": 37,
    }

    code = colors.get(color.lower(), 37)  # 기본값: white
    style = "1" if bold else "0"  # 1=bold, 0=normal

    print(f"\033[{style};{code}m{text}\033[0m")

def is_centered(
    center: tuple[int, int],
    img_size: tuple[int, int] = (1920, 1080),
    tol_px: int = 50
) -> tuple[bool, tuple[int, int], str]:
    """
    버튼 중심이 화면 중앙 근처인지 판정 (상하 무시, 이전 오차 무시).
    반환: (중앙에 있음?, 현재 오프셋(dx,dy), 다음 명령("left"/"right"/"stay"))
    """

    cx, cy = center
    W, H = img_size
    dx = cx - (W // 2)   # +면 오른쪽으로 치우침
    dy = cy - (H // 2)   # +면 아래로 치우침

    # 좌우 오차가 tol_px 이내면 stay
    if abs(dx) <= tol_px / 2:
        return True, (dx, dy), "stay"

    # dx>0 → 화면 오른쪽에 있음 → 오른쪽으로 회전
    if dx > 0:
        return False, (dx, dy), "right"
    else:
        return False, (dx, dy), "left"


def find_center_for_distance(fb_module, image_path, floor, ctx, out_name):
    buttons_raw = fb_module.find_buttons(['--image', image_path, '--no_vis', '--no_save'], ctx=ctx)
    buttons = parse_find_buttons_results(buttons_raw)
    target_button_bbox = get_bbox_by_text(buttons, floor)
    first_center = draw_bbox_and_center(image_path, target_button_bbox, out_name=out_name)

    return first_center