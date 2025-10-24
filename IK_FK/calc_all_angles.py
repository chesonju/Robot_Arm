from itertools import product
import numpy as np
import os
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"

def calc_positions(theta1, theta2, theta3, L=(104, 145, 180), base=(0,92)):
    """
    UI値(theta1~3, 0..180) → 内部角に変換 → 各点の(x,y)を返す。
    返り値は [(x0,y0),(x1,y1),(x2,y2),(x3,y3)] （ベース含む4点）
    """
    # UI → 内部角（あなたの既存ロジック踏襲）
    t1 = 180 - theta1
    t2 = 180 - theta2
    t3 =  90 - theta3
    thetas_deg = [t1, t2, t3]

    th = np.deg2rad(thetas_deg)
    x = [base[0]]; y = [base[1]]
    ang = 0.0
    for i in range(3):
        ang += th[i]
        x.append(x[-1] + L[i]*np.cos(ang))
        y.append(y[-1] + L[i]*np.sin(ang))
    return list(zip(x, y))

def all_angles_txt(L=(20,20,20), base=(0,10), step=1,
                   filename="angles_coords_step1.txt", buffer=100_000):
    """
    0..180（step刻み）全組合せの座標をTXTに保存。
    1行フォーマット:  t1_t2_t3_ (x0,y0),(x1,y1),(x2,y2),(x3,y3)
    tqdmで進捗バー表示。大容量なのでバッファ書き込み。
    """
    # 실행위치 기준으로 IK_FK 폴더 밑에 저장
    save_dir = HERE
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
    filepath = os.path.join(save_dir, filename)

    angles = range(0, 181, step)
    total = len(angles) ** 3

    with open(filepath, "w", encoding="utf-8") as f:
        buf = []
        for t1, t2, t3 in tqdm(product(angles, repeat=3),
                               total=total, desc=f"Writing coords (step={step})"):
            coords = calc_positions(t1, t2, t3, L, base)
            coords_str = ",".join(f"({x:.1f}, {y:.1f})" for x, y in coords)
            buf.append(f"{t1}_{t2}_{t3} {coords_str}\n")

            if len(buf) >= buffer:
                f.writelines(buf)
                buf.clear()

        if buf:
            f.writelines(buf)

    print(f"保存完了: {filepath}")

if __name__ == "__main__":
    # step=1（約 5,929,741 行）。時間＆サイズそれなりにデカいので覚悟を🙏
    all_angles_txt(L=(104, 145, 180), base=(0,92), step=1)
