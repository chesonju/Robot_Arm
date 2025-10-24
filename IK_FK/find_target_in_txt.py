import enhanced_plot
from collections import defaultdict
from tqdm import tqdm
import re
import os
import time
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"

def search_last_coord_in_file(filename, target_x, target_y, tol=1.0):
    """
    filename : 検索対象のテキストファイル
    target_x, target_y : 検索する座標 (float)
    tol : 許容範囲 ±tol
    """
    lines = []
    with open(filename, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # --- ぴったり一致のカウント ---
    exact_matches = []
    for line in lines:
        try:
            last_coord_str = line.split("),")[-1].strip()
            if not last_coord_str.endswith(")"):
                continue
            last_coord_str = last_coord_str.strip("()")
            x_str, y_str = last_coord_str.split(",")
            x_val = float(x_str.strip())
            y_val = float(y_str.strip())

            if x_val == target_x and y_val == target_y:
                exact_matches.append(line)
        except Exception:
            continue

    # --- 条件によって検索方法を切り替え ---
    if len(exact_matches) >= 5:
        # 範囲検索せず、ぴったり一致のみ返す
        return exact_matches
    else:
        # ±tol の範囲検索
        results = []
        for line in lines:
            try:
                last_coord_str = line.split("),")[-1].strip()
                if not last_coord_str.endswith(")"):
                    continue
                last_coord_str = last_coord_str.strip("()")
                x_str, y_str = last_coord_str.split(",")
                x_val = float(x_str.strip())
                y_val = float(y_str.strip())

                if abs(x_val - target_x) <= tol and abs(y_val - target_y) <= tol:
                    results.append(line)
            except Exception:
                continue
        return results
    
def filter_by_y_threshold(lines, threshold=5.0):
    """
    lines: 検索でヒットした行のリスト
    threshold: y座標の下限値（未満だと削除）
    """
    filtered = []
    for line in lines:
        try:
            # 最後の座標列を抽出
            coords_str = line.split(" ", 1)[1]  # "(0.0, 10.0),(...)"
            coords = coords_str.split("),")
            coords = [c.strip("() ") for c in coords]

            # x,y を float に変換
            coords = [tuple(map(float, c.split(","))) for c in coords]

            # (1,2,3) の y 値をチェック
            y_values = [coords[1][1], coords[2][1], coords[3][1]]
            if all(y >= threshold for y in y_values):
                filtered.append(line)
        except Exception:
            continue
    return filtered

def parse_thetas(line):
    """'122_22_140 (...)' -> (122.0, 22.0, 140.0)"""
    head = line.split(" ", 1)[0]
    t1, t2, t3 = map(float, head.split("_"))
    return t1, t2, t3

def find_most_central_line(lines,
                           ranges=((0,180),(0,180),(0,180)),
                           metric="l2"):
    """
    lines: 't1_t2_t3 (..coords..)' の文字列リスト
    ranges: ((t1_lo,t1_hi), (t2_lo,t2_hi), (t3_lo,t3_hi))
    metric: "l1" or "l2"（ど真ん中からの距離評価）

    戻り: (best_line, best_thetas_tuple, score)
          該当なしなら (None, None, None)
    """
    # 中央値（ど真ん中）を作る
    mids = tuple((lo+hi)/2.0 for (lo,hi) in ranges)

    def in_range(t, rng):
        lo, hi = rng
        return lo <= t <= hi

    def score_fn(th):
        # ど真ん中との差（L1 or L2）
        diffs = [abs(th[i] - mids[i]) for i in range(3)]
        if metric == "l1":
            return sum(diffs)
        else:  # L2
            return sum(d*d for d in diffs) ** 0.5

    best = (None, None, None)  # line, thetas, score
    for line in lines:
        try:
            th = parse_thetas(line)
        except Exception:
            continue
        # レンジ内チェック（全部入ってるやつだけ評価）
        if not all(in_range(th[i], ranges[i]) for i in range(3)):
            continue
        sc = score_fn(th)
        if best[2] is None or sc < best[2]:
            best = (line, th, sc)

    return best

def find_target_in_file(target_x, target_y, filename, ranges=((0,180),(0,180),(0,180)), plot=False):
    hits = search_last_coord_in_file(filename, target_x, target_y, tol=1.0)

    # print(hits)

    # y閾値でフィルタ
    filtered_hits = filter_by_y_threshold(hits, threshold=30)

    # for h in filtered_hits:
        # print(h)
        #　t1, t2, t3 = parse_thetas(h)
        # enhanced_plot.run_arm_gui(theta1=t1, theta2=t2, theta3=t3, just_save_plot=True)

    # デフォ: 全θ 0..180 の中点(=90)に一番近いやつ

    if filtered_hits:
        best_line, (t1, t2, t3), score = find_most_central_line(filtered_hits, ranges=ranges)
        # print(best_line, (t1, t2, t3), score)

        if plot:
            enhanced_plot.run_arm_gui(theta1=t1, theta2=t2, theta3=t3, just_save_plot=True)
        
        return best_line
    
    else:
        return None

def _parse_last_xy(line: str):
    """
    1行から最後の座標 (x, y) を取り出して返す。
    フォーマット例: '...,(-25.0, 27.0)'
    """
    try:
        tail = line.rsplit("),", 1)[-1].strip()
        if not tail.endswith(")"):
            return None
        x_str, y_str = tail.strip("()").split(",")
        return float(x_str.strip()), float(y_str.strip())
    except Exception:
        return None

def _key_eq(v: float, tol_equal: float) -> int:
    """
    “同じ”扱いのためのバケツキー。tol_equal の幅で丸め込む。
    例: tol_equal=1e-3 なら 0.001 単位で同一キー。
    """
    return round(v / tol_equal)

def find_axis_jump_pairs(filename: str, move=50.0, tol_equal=1e-3, tol_move=1e-2,
                         y_min=0.0, show_progress=True):
    """
    ファイル内の“最後の座標(x, y)”を全行から抽出して、
    - x が同じ(±tol_equal) かつ |Δy| ≈ move(±tol_move)
    - y が同じ(±tol_equal) かつ |Δx| ≈ move(±tol_move)
    のペアを列挙。最後に y>=y_min でフィルタ。
    """
    def _parse_last_xy(line: str):
        try:
            tail = line.rsplit("),", 1)[-1].strip()
            if not tail.endswith(")"):
                return None
            x_str, y_str = tail.strip("()").split(",")
            return float(x_str.strip()), float(y_str.strip())
        except Exception:
            return None

    def _key_eq(v: float, tol: float) -> int:
        return round(v / tol)

    # --- 1) 全行読み込み＆座標抽出 ---
    entries = []  # (idx, line, x, y)
    raw_lines = [ln.strip() for ln in open(filename, encoding="utf-8") if ln.strip()]
    pbar = tqdm(total=len(raw_lines), desc="読み込み＆座標抽出") if show_progress else None
    for idx, line in enumerate(raw_lines):
        xy = _parse_last_xy(line)
        if xy is not None:
            x, y = xy
            entries.append((idx, line, x, y))
        if pbar:
            pbar.update(1)
    if pbar: pbar.close()

    # --- 2) x固定で Δy 探索 ---
    by_x = defaultdict(list)
    for idx, line, x, y in entries:
        by_x[_key_eq(x, tol_equal)].append((idx, line, x, y))

    same_x_y_move = []
    total_i_x = sum(len(bucket) for bucket in by_x.values())
    pbar = tqdm(total=total_i_x, desc="x固定で Δy 探索") if show_progress else None
    for bucket in by_x.values():
        bucket.sort(key=lambda t: t[3])  # yでソート
        n = len(bucket)
        j = 0
        for i in range(n):
            x_i, y_i = bucket[i][2], bucket[i][3]
            while j < n and bucket[j][3] - y_i < move - tol_move:
                j += 1
            k = j
            while k < n and bucket[k][3] - y_i <= move + tol_move:
                dy = bucket[k][3] - y_i
                if abs(dy - move) <= tol_move:
                    irec, jrec = bucket[i], bucket[k]
                    same_x_y_move.append({
                        "i": irec[0], "j": jrec[0],
                        "x": x_i, "y1": y_i, "y2": jrec[3],
                        "line_i": irec[1], "line_j": jrec[1],
                    })
                k += 1
            if pbar:
                pbar.update(1)
    if pbar: pbar.close()

    # --- 3) y固定で Δx 探索 ---
    by_y = defaultdict(list)
    for idx, line, x, y in entries:
        by_y[_key_eq(y, tol_equal)].append((idx, line, x, y))

    same_y_x_move = []
    total_i_y = sum(len(bucket) for bucket in by_y.values())
    pbar = tqdm(total=total_i_y, desc="y固定で Δx 探索") if show_progress else None
    for bucket in by_y.values():
        bucket.sort(key=lambda t: t[2])  # xでソート
        n = len(bucket)
        j = 0
        for i in range(n):
            x_i, y_i = bucket[i][2], bucket[i][3]
            while j < n and bucket[j][2] - x_i < move - tol_move:
                j += 1
            k = j
            while k < n and bucket[k][2] - x_i <= move + tol_move:
                dx = bucket[k][2] - x_i
                if abs(dx - move) <= tol_move:
                    irec, jrec = bucket[i], bucket[k]
                    same_y_x_move.append({
                        "i": irec[0], "j": jrec[0],
                        "y": y_i, "x1": x_i, "x2": jrec[2],
                        "line_i": irec[1], "line_j": jrec[1],
                    })
                k += 1
            if pbar:
                pbar.update(1)
    if pbar: pbar.close()

    # --- 4) y >= y_min でフィルタ ---
    same_x_y_move = [p for p in same_x_y_move if p["y1"] >= y_min and p["y2"] >= y_min]
    same_y_x_move = [p for p in same_y_x_move if p["y"] >= y_min]

    return {
        "same_x_y_move": same_x_y_move,
        "same_y_x_move": same_y_x_move,
    }

def find_axis_jump_pairs_to_txt(filename=HERE + "angles_coords_step1.txt", move=50):
    '''
    사진으로 거리 계산용 (x 고정 & y 변화) 및 (y 고정 & x 변화) 쌍을 찾아서 텍스트로 저장, move 기본 50
    '''
    res = find_axis_jump_pairs(filename, move=move, tol_equal=1e-3, tol_move=1e-2, show_progress=True)
    output = f"{HERE}{move}_jump_result.txt"
    with open(output, "w", encoding="utf-8") as f:
        # --- xが同じ & yが±move ---
        f.write(f"xが同じ & yが±{move}: {len(res['same_x_y_move'])}\n")
        for p in res["same_x_y_move"]:
            f.write(f"{p['line_i']} ⇨ {p['line_j']}\n")
        
        f.write("\n")
        # --- yが同じ & xが±move ---
        f.write(f"yが同じ & xが±{move}: {len(res['same_y_x_move'])}\n")
        for p in res["same_y_x_move"]:
            f.write(f"{p['line_i']} ⇨ {p['line_j']}\n")

    print(f"結果を {output} に保存しました。")

def find_target_in_file_to_png(target_x=41.0, target_y=367.0, ranges=((0,175),(0,175),(0,175))):
    result = find_target_in_file(target_x=target_x, target_y=target_y, filename=HERE+"angles_coords_step1.txt", ranges=ranges)

    if result:
        print(result)
        t1, t2, t3 = parse_thetas(result)
        enhanced_plot.run_arm_gui(theta1=t1, theta2=t2, theta3=t3, just_save_plot=True)

    else:
        print("該当する座標が見つかりませんでした。")

def filter_jump_result_same_last_y_and_save_txt(input_path: str, output_path: str = None, tol_equal: float = 1e-6):
    """
    *_jump_result.txt の中で、
    左右それぞれの「最後の座標」と「その前の座標」の y 値が tol_equal 以内で同じペアだけ抽出。

    Parameters
    ----------
    input_path : str
        入力ファイルパス（例: '50_jump_result.txt'）
    output_path : str
        保存先（省略時は '<input>_same_last_y.txt'）
    tol_equal : float
        yが同じとみなす許容誤差
    """

    def parse_all_xy(line: str):
        """'169_40_177 (0.0, 92.0),(...),(...),(...)' → [(x0,y0), ...]"""
        try:
            coords_str = line.split(" ", 1)[1]
        except IndexError:
            return None
        parts = [p.strip().rstrip(")") for p in coords_str.split("),")]
        out = []
        for p in parts:
            p = p.lstrip("(").strip()
            if not p:
                continue
            xs, ys = p.split(",")
            out.append((float(xs.strip()), float(ys.strip())))
        return out if out else None

    if output_path is None:
        output_path = input_path.rsplit(".", 1)[0] + "_same_last_y.txt"

    hits = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "⇨" not in line:
                continue
            left, right = line.split("⇨", 1)
            left_pts = parse_all_xy(left.strip())
            right_pts = parse_all_xy(right.strip())
            if not left_pts or not right_pts:
                continue
            # 最後とその前のy差を確認
            y_last_L, y_prev_L = left_pts[-1][1], left_pts[-2][1]
            y_last_R, y_prev_R = right_pts[-1][1], right_pts[-2][1]
            if (abs(y_last_L - y_prev_L) <= tol_equal) and (abs(y_last_R - y_prev_R) <= tol_equal):
                hits.append(line)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"# same_last_y pairs (tol_equal={tol_equal})\n")
        out.write(f"# hits={len(hits)}\n\n")
        for h in hits:
            out.write(h + "\n")

    print(f"yが同じペア: {len(hits)} 件")
    print(f"保存先: {output_path}")
    return hits, output_path

PAIR_RE = re.compile(
    r"""^\s*
        (?P<a1>\d+)_(?P<a2>\d+)_(?P<a3>\d+)\s+
        (?P<a_coords>\([^)]+\)(?:,\([^)]+\))*)
        \s*⇨\s*
        (?P<b1>\d+)_(?P<b2>\d+)_(?P<b3>\d+)\s+
        (?P<b_coords>\([^)]+\)(?:,\([^)]+\))*)
        \s*$
    """, re.X
)

def expected_png_name(angles: str, coords: str) -> str:
    # txtの行と完全一致するファイル名を期待
    return f"{HERE}{angles} {coords}.png"

def find_png_by_prefix(angles: str) -> str | None:
    # 念のためのフォールバック：座標が微妙に違っても angles 先頭一致で最新を拾う
    candidates = sorted(glob.glob(f"{angles} *.png"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None

def wait_for_file(path: str, timeout=3.0):
    # 生成完了をちょい待ち（最大3秒）
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return os.path.exists(path)

def stitch_side_by_side(left_path: str, right_path: str, out_path: str, title: str | None = None):
    left = Image.open(left_path).convert("RGBA")
    right = Image.open(right_path).convert("RGBA")

    pad = 20  # 画像の間と外枠
    title_h = 60 if title else 0

    W = left.width + right.width + pad * 3
    H = max(left.height, right.height) + pad * 2 + title_h
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    x_left = pad
    y_left = pad + title_h
    canvas.paste(left, (x_left, y_left), left)

    x_right = x_left + left.width + pad
    y_right = pad + title_h
    canvas.paste(right, (x_right, y_right), right)

    # 仕切り線
    draw = ImageDraw.Draw(canvas)
    mid_x = x_right - int(pad/2)
    draw.line([(mid_x, pad + title_h), (mid_x, H - pad)], fill=(0,0,0,64), width=2)

    # タイトル（フォントなくてもOKなデフォ描画）
    if title:
        draw.text((pad, 10), title, fill=(0,0,0,255))

    canvas.convert("RGB").save(out_path, format="PNG")

def combine_pair_txt_to_side_png(txt_path: str, out_dir: str = "combined_out", cleanup: bool = True):
    '''
    찾은 페어 시각화 함수
    '''
    os.makedirs(out_dir, exist_ok=True)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            m = PAIR_RE.match(line)
            if not m:
                print(f"[skip L{line_no}] 解析できない行: {line}")
                continue

            a_angles = f"{m['a1']}_{m['a2']}_{m['a3']}"
            b_angles = f"{m['b1']}_{m['b2']}_{m['b3']}"
            a_coords = m['a_coords']
            b_coords = m['b_coords']

            a1, a2, a3 = int(m['a1']), int(m['a2']), int(m['a3'])
            b1, b2, b3 = int(m['b1']), int(m['b2']), int(m['b3'])

            print(f"[L{line_no}] A={a_angles} ⇨ B={b_angles}")

            # それぞれ描画して保存（enhanced_plot 側で保存される想定）
            enhanced_plot.run_arm_gui(theta1=a1, theta2=a2, theta3=a3, just_save_plot=True)
            enhanced_plot.run_arm_gui(theta1=b1, theta2=b2, theta3=b3, just_save_plot=True)

            # 期待ファイル名を構築
            a_png = expected_png_name(a_angles, a_coords)
            b_png = expected_png_name(b_angles, b_coords)

            # 生成待ち + フォールバック
            if not wait_for_file(a_png):
                alt = find_png_by_prefix(a_angles)
                if alt:
                    a_png = alt
            if not wait_for_file(b_png):
                alt = find_png_by_prefix(b_angles)
                if alt:
                    b_png = alt

            if not os.path.exists(a_png) or not os.path.exists(b_png):
                print(f"  → 画像が見つからないためスキップ")
                continue

            out_name = f"{a_angles}⇨{b_angles}.png"
            out_path = os.path.join(out_dir, out_name)
            title = f"{a_angles}  →  {b_angles}"

            # 合成＆削除処理
            try:
                stitch_side_by_side(a_png, b_png, out_path, title=title)
                print(f"  → {out_path} を出力👌")

                if cleanup:
                    for p in {a_png, b_png}:  # 同じパスなら1回だけ
                        try:
                            os.remove(p)
                            print(f"    - cleanup: {p} 削除済み")
                        except FileNotFoundError:
                            pass
                        except OSError as e:
                            print(f"    - cleanup失敗: {p} ({e})")
            except Exception as e:
                print(f"  → 合成に失敗: {e}（元画像は残す）")

def _rgb_to_hsv_np(rgb):  # rgb: uint8 array (...,3)
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    v = maxc
    d = maxc - minc
    s = np.where(maxc == 0, 0, d / (maxc + 1e-6))

    # hue
    h = np.zeros_like(maxc)
    mask = d > 1e-6
    rc = np.zeros_like(maxc); gc = np.zeros_like(maxc); bc = np.zeros_like(maxc)
    rc[mask] = ((maxc - r) / (d + 1e-6))[mask]
    gc[mask] = ((maxc - g) / (d + 1e-6))[mask]
    bc[mask] = ((maxc - b) / (d + 1e-6))[mask]

    mr = (maxc == r) & mask
    mg = (maxc == g) & mask
    mb = (maxc == b) & mask
    h[mr] = (bc - gc)[mr]
    h[mg] = 2.0 + (rc - bc)[mg]
    h[mb] = 4.0 + (gc - rc)[mb]
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)

def _hsv_to_rgb_np(hsv):  # hsv floats in [0,1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(np.int32)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = i % 6

    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def recolor_indigo_to_pink(pil_img,
                           indigo_hue_deg=(218, 255),
                           min_sat=0.25,
                           min_val=0.25,
                           dest_hue_deg=330,
                           dest_sat=0.85):
    """
    2枚目用：藍色(インディゴ系のHue帯)に該当するピクセルの色相をピンクへ置換。
    明るさVは維持、彩度はdest_satで上書き（元のSが低ければ上げる）。
    """
    img = pil_img.convert("RGBA")
    rgb = np.array(img)[..., :3]
    hsv = _rgb_to_hsv_np(rgb)

    # 条件: Hueが indigo_hue_deg 範囲 && S, V がそこそこある
    h_deg = (hsv[..., 0] * 360.0)
    s = hsv[..., 1]
    v = hsv[..., 2]

    deg_lo, deg_hi = indigo_hue_deg
    mask = (h_deg >= deg_lo) & (h_deg <= deg_hi) & (s >= min_sat) & (v >= min_val)

    # 置換：Hueだけピンクへ、彩度は上げ気味、明るさは維持
    hsv2 = hsv.copy()
    hsv2[..., 0][mask] = (dest_hue_deg % 360) / 360.0
    hsv2[..., 1][mask] = np.maximum(hsv2[..., 1][mask], dest_sat)

    recolored_rgb = _hsv_to_rgb_np(hsv2)
    out = np.array(img).copy()
    out[..., :3] = recolored_rgb
    return Image.fromarray(out, mode="RGBA")

def set_opacity(img: Image.Image, alpha: float) -> Image.Image:
    """0.0~1.0 の不透明度を掛ける"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    r, g, b, a = img.split()
    a = a.point(lambda p: int(p * alpha))
    return Image.merge("RGBA", (r, g, b, a))

def hue_shift_image(pil_img: Image.Image, hue_deg: float = 25.0, sat_gain: float = 1.0):
    """
    画像全体の色相を hue_deg (度) だけ回転。彩度は sat_gain 倍。
    例: hue_deg=+25 で全体を少し赤寄りに、-25 で青寄りに。
    """
    img = pil_img.convert("RGBA")
    arr = np.array(img)
    rgb = arr[..., :3].astype(np.float32)

    # RGB -> HSV
    hsv = _rgb_to_hsv_np(rgb.astype(np.uint8))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 色相回転 + 彩度ゲイン
    h = (h + hue_deg / 360.0) % 1.0
    s = np.clip(s * sat_gain, 0.0, 1.0)

    # HSV -> RGB
    hsv2 = np.stack([h, s, v], axis=-1)
    rgb2 = _hsv_to_rgb_np(hsv2)

    out = arr.copy()
    out[..., :3] = rgb2
    return Image.fromarray(out, mode="RGBA")

def overlay_images(
    img1_path,
    img2_path,
    out_path,
    opacity1=1.0,          # 1枚目の不透明度
    opacity2=0.6,          # 2枚目の不透明度
    recolor_mode="global_shift",  # "global_shift" | "indigo_to_pink" | None
    hue_deg=25.0,
    sat_gain=1.05
):
    im1 = Image.open(img1_path).convert("RGBA")
    im2 = Image.open(img2_path).convert("RGBA")

    # サイズ揃え
    if im2.size != im1.size:
        im2 = im2.resize(im1.size, resample=Image.BICUBIC)

    # 2枚目の色処理
    if recolor_mode == "global_shift":
        im2 = hue_shift_image(im2, hue_deg=hue_deg, sat_gain=sat_gain)
    elif recolor_mode == "indigo_to_pink":
        im2 = recolor_indigo_to_pink(im2)

    # 不透明度変更
    im1a = set_opacity(im1, opacity1)
    im2a = set_opacity(im2, opacity2)

    # 完全透明キャンバスに合成
    base = Image.new("RGBA", im1a.size, (255, 255, 255, 0))
    out = Image.alpha_composite(base, im1a)
    out = Image.alpha_composite(out, im2a)

    # 背景透明のまま保存
    out.save(out_path, format="PNG")

def combine_pair_txt_to_overlay_png(txt_path: str,
                out_dir: str = "combined_out",
                cleanup: bool = True,
                mode: str = "overlay",  # "overlay" or "side_by_side"
                opacity1: float = 1.0,
                opacity2: float = 0.5,
                recolor_mode: str = "global_shift",  # "global_shift" | "indigo_to_pink" | None
                hue_deg: float = 28.0,
                sat_gain: float = 1.08):
    '''
    찾은 페어 시각화용 함수
    '''
    os.makedirs(out_dir, exist_ok=True)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            m = PAIR_RE.match(line)
            if not m:
                print(f"[skip L{line_no}] 解析できない行: {line}")
                continue

            a_angles = f"{m['a1']}_{m['a2']}_{m['a3']}"
            b_angles = f"{m['b1']}_{m['b2']}_{m['b3']}"
            a_coords = m['a_coords']
            b_coords = m['b_coords']

            a1, a2, a3 = int(m['a1']), int(m['a2']), int(m['a3'])
            b1, b2, b3 = int(m['b1']), int(m['b2']), int(m['b3'])

            print(f"[L{line_no}] A={a_angles} ⇨ B={b_angles}")

            enhanced_plot.run_arm_gui(theta1=a1, theta2=a2, theta3=a3, just_save_plot=True)
            enhanced_plot.run_arm_gui(theta1=b1, theta2=b2, theta3=b3, just_save_plot=True)

            a_png = expected_png_name(a_angles, a_coords)
            b_png = expected_png_name(b_angles, b_coords)

            if not wait_for_file(a_png):
                alt = find_png_by_prefix(a_angles)
                if alt: a_png = alt
            if not wait_for_file(b_png):
                alt = find_png_by_prefix(b_angles)
                if alt: b_png = alt

            if not os.path.exists(a_png) or not os.path.exists(b_png):
                print("  → 画像が見つからないためスキップ")
                continue

            out_name = f"{a_angles}⇨{b_angles}{'_overlay' if mode=='overlay' else ''}.png"
            out_path = os.path.join(out_dir, out_name)

            try:
                if mode == "overlay":
                    overlay_images(
                        a_png, b_png, out_path,
                        opacity1=opacity1, opacity2=opacity2,
                        recolor_mode=recolor_mode,
                        hue_deg=hue_deg,
                        sat_gain=sat_gain
                    )
                else:
                    title = f"{a_angles}  →  {b_angles}"
                    stitch_side_by_side(a_png, b_png, out_path, title=title)

                print(f"  → {out_path} を出力👌")
            except Exception as e:
                print(f"  → 合成失敗: {e}（元画像は残す）")
                continue
            finally:
                # 合成の成功失敗に関わらず cleanup
                if cleanup:
                    for p in {a_png, b_png}:
                        try:
                            os.remove(p)
                            print(f"    - cleanup: {p} 削除済み")
                        except FileNotFoundError:
                            pass
                        except OSError as e:
                            print(f"    - cleanup失敗: {p} ({e})")

if __name__ == "__main__":
    # 一定数の違いがある座標を探す(ここは50)
    find_axis_jump_pairs_to_txt()

    # 特定の座標を探して、対応する角度を描画
    # find_target_in_file_to_png(-300, 200)

    # y座標が同じベアを探して保存
    filter_jump_result_same_last_y_and_save_txt(input_path=f"{HERE}50_jump_result.txt",tol_equal=1e-6)

    # 横にならべて合成
    # combine_pair_txt_to_side_png(txt_path=f"{HERE}50_jump_result_same_last_y.txt", out_dir=f"{HERE}combined_out_side", cleanup=True)

    # オーバーレイ合成
    combine_pair_txt_to_overlay_png(txt_path=f"{HERE}50_jump_result_same_last_y.txt", out_dir=f"{HERE}combined_out_overlay")