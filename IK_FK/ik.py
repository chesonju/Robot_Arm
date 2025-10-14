import math
import numpy as np
import enhanced_plot
# ---------- 共通: 1つのφで3R IKを解く ----------
def _ik_3r_with_phi(x, y, phi_deg, L, base, elbow):
    L1, L2, L3 = L
    xb, yb = base

    # ベース原点へ並進
    xt, yt = x - xb, y - yb
    phi = math.radians(phi_deg)

    # 手首位置（先端からL3戻す）
    xw = xt - L3 * math.cos(phi)
    yw = yt - L3 * math.sin(phi)

    r2 = xw*xw + yw*yw
    r = math.sqrt(r2)

    # 到達判定（2R部分）
    reach_min = abs(L1 - L2)
    reach_max = L1 + L2
    if r < reach_min - 1e-9 or r > reach_max + 1e-9:
        return None  # 解なし

    # 2R: θ2
    cos_t2 = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2 = math.acos(cos_t2)
    if elbow == "down":
        t2 = -t2

    # 2R: θ1
    k1 = L1 + L2*math.cos(t2)
    k2 = L2*math.sin(t2)
    t1 = math.atan2(yw, xw) - math.atan2(k2, k1)

    # θ3 は φ = θ1 + θ2 + θ3 を満たす
    t3 = math.radians(phi_deg) - (t1 + t2)

    th = (math.degrees(t1), math.degrees(t2), math.degrees(t3))
    return th

def _theta_to_ui(th):
    th1, th2, th3 = th
    return (180.0 - th1, 180.0 - th2, 90.0 - th3)

def _angle_wrap(a):
    """-180..180へ正規化"""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def _cost_delta(th, ref_th=None, ref_phi=None):
    """
    解(th)が参照にどれだけ近いかのコスト。
    - ref_th があれば 角度差（ラップあり）のL2
    - ref_phi しかなければ 末端姿勢(θ1+θ2+θ3) と ref_phi の角度差
    - 参照が無ければ 0（=どれでもOK）
    """
    t1, t2, t3 = th
    if ref_th is not None:
        d1 = _angle_wrap(t1 - ref_th[0])
        d2 = _angle_wrap(t2 - ref_th[1])
        d3 = _angle_wrap(t3 - ref_th[2])
        return d1*d1 + d2*d2 + d3*d3
    if ref_phi is not None:
        phi = _angle_wrap((t1 + t2 + t3) - ref_phi)
        return phi*phi
    return 0.0

def ik_planar_3r_auto(
    x, y,
    phi_deg=None,                  # 指定あれば優先
    L=(20,20,20),
    base=(0,10),
    elbow_opts=("up","down"),      # 両肘を候補に
    ref_theta=None,                # 参照: 前フレームの内部角(θ1,θ2,θ3) [deg]
    ref_phi=None,                  # 参照: 前フレームの先端姿勢φ [deg]
    coarse_step=15,                # φ自動探索の粗スキャン刻み
    refine_halfwin=10,             # 近傍微調整の±範囲
    refine_step=1,                 # 微調整刻み
    return_ui=True
):
    """
    φ未指定でも “それっぽい” 姿勢を自動決定する3R IK。
    - 優先: phi_deg 指定
    - 次点: ref_theta/ref_phi に近い解になるφを探索
    - 参照無いとき: ベース→目標方向を向くφを初期候補に

    戻り:
      {'theta': (θ1,θ2,θ3), 'theta_ui': (...), 'phi': 採用したφ}
    """
    # 1) φが指定されていればそのまま解く
    if phi_deg is not None:
        best = None
        for elbow in elbow_opts:
            th = _ik_3r_with_phi(x, y, phi_deg, L, base, elbow)
            if th is None:
                continue
            c = _cost_delta(th, ref_theta, ref_phi)
            if best is None or c < best[0]:
                best = (c, th, elbow, phi_deg)
        if best is None:
            raise ValueError("到達不可（指定φでは解なし）")
        result = {'theta': best[1], 'phi': best[3]}
        if return_ui:
            result['theta_ui'] = _theta_to_ui(best[1])
        return result

    # 2) φ未指定 → 参照に近いφを探索
    #    まず初期候補φ0を決める
    if ref_phi is not None:
        phi0 = ref_phi
    elif ref_theta is not None:
        phi0 = sum(ref_theta)
    else:
        # 参照なし → ベース→目標方向
        xb, yb = base
        phi0 = math.degrees(math.atan2(y - yb, x - xb))

    # 粗スキャン: phi0±180 を coarse_step で
    cand_phis = list(range(int(phi0 - 180), int(phi0 + 181), coarse_step))
    # 微調整: 後でbest近傍を細かく

    best = None
    for phi_cand in cand_phis:
        for elbow in elbow_opts:
            th = _ik_3r_with_phi(x, y, phi_cand, L, base, elbow)
            if th is None:
                continue
            c = _cost_delta(th, ref_theta, ref_phi)
            if best is None or c < best[0]:
                best = (c, th, elbow, phi_cand)

    if best is None:
        raise ValueError("到達不可（粗探索で解なし）")

    # 近傍リファイン
    phi_center = best[3]
    fine_phis = range(int(phi_center - refine_halfwin),
                      int(phi_center + refine_halfwin + 1),
                      refine_step)

    for phi_cand in fine_phis:
        for elbow in elbow_opts:
            th = _ik_3r_with_phi(x, y, phi_cand, L, base, elbow)
            if th is None:
                continue
            c = _cost_delta(th, ref_theta, ref_phi)
            if c < best[0]:
                best = (c, th, elbow, phi_cand)

    result = {'theta': best[1], 'phi': _angle_wrap(best[3])}
    if return_ui:
        result['theta_ui'] = _theta_to_ui(best[1])
    return result


# ---- 使い方デモ ----
if __name__ == "__main__":
    L = (104,145,150); base=(0,92)

    # 参照があるケース（前フレームの関節角に近い解を優先）
    # ref = (10.0, -20.0, 5.0)  # 例: 前フレーム内部角
    # sol = ik_planar_3r_auto(40, 25, phi_deg=None, L=L, base=base, ref_theta=ref)
    # print("auto(参照θあり):", sol)

    # 参照がないケース（ベース→目標方向を向く）
    sol2 = ik_planar_3r_auto(-300, 200, phi_deg=None, L=L, base=base)

    print("auto(参照なし):", sol2)
    enhanced_plot.run_arm_gui(
        theta1=round(sol2['theta_ui'][0]),
        theta2=round(sol2['theta_ui'][1]),
        theta3=round(sol2['theta_ui'][2]),
        L=L, base=base, just_save_plot=True
    )

    # φを明示するケース（従来どおり）
    # sol3 = ik_planar_3r_auto(60, 10, phi_deg=0, L=L, base=base)
    # print("phi=0指定:", sol3)
