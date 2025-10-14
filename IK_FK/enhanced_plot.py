# enhanced_plot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Wedge

def run_arm_gui(
    theta1=90, theta2=90, theta3=90,
    slider_step=1,
    sector_alpha=0.15,
    point_size=4,
    L=(104, 145, 180),
    base=(0, 92),
    joint_limits=((0, -180), (0, -180), (-90, 90)),
    joint_zero_offset=(0, 0, 0),
    show_origin=True,
    block=False,                 # ← ここ重要：Falseで非ブロッキング表示
    just_save_plot=False,        # Trueならプロットを保存して終了
):
    if len(L) != 3: raise ValueError("L は長さ3のタプル/リストにしてね")
    if len(joint_limits) != 3 or len(joint_zero_offset) != 3:
        raise ValueError("joint_limits / joint_zero_offset は3要素で。")

    x_base, y_base = base

    def ui_to_internal(t1_ui, t2_ui, t3_ui):
        return [180 - t1_ui, 180 - t2_ui, 90 - t3_ui]

    def calc_positions(thetas_deg):
        thetas_rad = np.deg2rad(thetas_deg)
        x = [x_base]; y = [y_base]; ang = 0.0
        for i in range(3):
            ang += thetas_rad[i]
            x.append(x[-1] + L[i]*np.cos(ang))
            y.append(y[-1] + L[i]*np.sin(ang))
        return x, y

    def joint_radius(i): return L[i]

    def nice_axes_limits():
        reach = sum(L); m = max(10, reach*0.15)
        return (x_base - reach - m, x_base + reach + m,
                y_base - m,        y_base + reach + m)

    theta_init = [theta1, theta2, theta3]
    x0, y0 = calc_positions(ui_to_internal(*theta_init))

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    line, = ax.plot(x0, y0, marker='o', markersize=point_size, linewidth=3)
    start_point, = ax.plot(x0[0], y0[0], 'ko', markersize=point_size, label='Base')
    if show_origin:
        ax.plot(0, 0, 'ro', markersize=point_size, label='(0, 0)')

    # sliders
    left, width, h = 0.2, 0.6, 0.03
    ax_t1 = plt.axes([left, 0.25, width, h])
    ax_t2 = plt.axes([left, 0.19, width, h])
    ax_t3 = plt.axes([left, 0.13, width, h])
    s1 = Slider(ax_t1, 'Theta1', 0, 180, valinit=theta_init[0], valstep=slider_step)
    s2 = Slider(ax_t2, 'Theta2', 0, 180, valinit=theta_init[1], valstep=slider_step)
    s3 = Slider(ax_t3, 'Theta3', 0, 180, valinit=theta_init[2], valstep=slider_step)

    # coordinate labels
    coord_texts = []
    for xi, yi in zip(x0, y0):
        coord_texts.append(ax.text(xi, yi+2, f"({xi:.1f}, {yi:.1f})", ha='center', fontsize=8))

    # wedges
    sector_colors = ['red', 'green', 'blue']
    sector_patches = []
    for i in range(3):
        w = Wedge((x0[i], y0[i]), joint_radius(i), 0, 0,
                  alpha=sector_alpha, facecolor=sector_colors[i], edgecolor='black')
        w.set_linewidth(0.8)
        ax.add_patch(w); sector_patches.append(w)

    def update_sectors(jx, jy, thetas_deg):
        th = np.deg2rad(thetas_deg)
        bases = [0.0, th[0], th[0]+th[1]]
        for i in range(3):
            base_deg = np.rad2deg(bases[i]); off = joint_zero_offset[i]
            amin, amax = joint_limits[i]
            w = sector_patches[i]
            w.set_center((jx[i], jy[i]))
            w.set_radius(joint_radius(i))
            w.set_theta1(base_deg + off + amin)
            w.set_theta2(base_deg + off + amax)

    def _update_from_ui():
        thetas = ui_to_internal(s1.val, s2.val, s3.val)
        X, Y = calc_positions(thetas)
        line.set_xdata(X); line.set_ydata(Y); start_point.set_data([X[0]], [Y[0]])
        for t, xi, yi in zip(coord_texts, X, Y):
            t.set_position((xi, yi+2)); t.set_text(f"({xi:.1f}, {yi:.1f})")
        update_sectors(X, Y, thetas)
        fig.canvas.draw_idle()

    s1.on_changed(lambda _: _update_from_ui())
    s2.on_changed(lambda _: _update_from_ui())
    s3.on_changed(lambda _: _update_from_ui())
    _update_from_ui()

    xmin, xmax, ymin, ymax = nice_axes_limits()
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box'); ax.grid(True)
    ax.set_title('3-Joint Arm - Angle Control'); ax.legend(loc='upper left')

    if just_save_plot:
        # 角度と座標テキストをファイル名に入れる（長すぎるなら整形・短縮も可）
        coords_str = ",".join(t.get_text() for t in coord_texts)
        fname = f"./IK_FK/{int(theta1)}_{int(theta2)}_{int(theta3)} {coords_str}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fname

    class ArmGUIController:
        def __init__(self):
            self.fig = fig; self.ax = ax
            self.s1 = s1; self.s2 = s2; self.s3 = s3
        # 外部からUI角度で更新（0..180）
        def set_ui_angles(self, t1_ui, t2_ui, t3_ui):
            self.s1.set_val(float(t1_ui))
            self.s2.set_val(float(t2_ui))
            self.s3.set_val(float(t3_ui))
        # 外部から内部角で更新
        def set_internal_angles(self, th1, th2, th3):
            self.set_ui_angles(180.0 - th1, 180.0 - th2, 90.0 - th3)
        def get_ui_angles(self):
            return (self.s1.val, self.s2.val, self.s3.val)
        def get_internal_angles(self):
            return (180.0 - self.s1.val, 180.0 - self.s2.val, 90.0 - self.s3.val)

    ctrl = ArmGUIController()
    plt.show(block=block)  # block=False ならすぐ戻る
    return ctrl

if __name__ == "__main__":    
    # 使い方例
    # run_arm_gui(theta1=159, theta2=79, theta3=32, just_save_plot=True)
    # run_arm_gui(theta1=134, theta2=115, theta3=21, just_save_plot=True)

    gui = run_arm_gui(theta1=70, theta2=120, theta3=40, block=True)  # 非ブロッキング
