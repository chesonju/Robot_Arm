import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# -----------------------------------------------------------------
# 1. Matplotlib setup (for minus sign)
# -----------------------------------------------------------------
plt.rcParams['axes.unicode_minus'] = False  # Prevents minus sign issues

# -----------------------------------------------------------------
# 2. Main visualization function (수정됨)
# -----------------------------------------------------------------
def plot_geometry(x1, y1, length, angle_input, show_plot=False, save_path=None):
    """
    Plots geometry based on given coordinates and vectors.
    
    Args:
        x1 (float): X coordinate for mark1
        y1 (float): Y coordinate for mark1
        length (float): Length of the vector from mark2
        angle_input (float): Angle from the x- negative direction (180 deg) at mark2
        show_plot (bool): If True, display the plot window (plt.show()).
        save_path (str or None): If provided, save the plot to this file path.
        
    Returns:
        tuple (float, float): The (x, y) coordinates of the vector's end point.
    """
    
    # ---------------------------------
    # 2-1. Calculate mark2 coordinates
    # ---------------------------------
    x2 = x1 + 85
    y2 = y1 + 55

    # ---------------------------------
    # 2-2. Setup the plot
    # ---------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-600, 50)
    ax.set_ylim(-50, 600)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Coordinate Visualization")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.grid(True, linestyle=':', alpha=0.7)

    # ---------------------------------
    # 2-3. Plot points (mark1, mark2)
    # ---------------------------------
    ax.plot(x1, y1, 'bo', markersize=8, label=f'Gripper End ({x1:.1f}, {y1:.1f})')
    ax.text(x1, y1 + 5, f' Gripper End ({x1:.1f}, {y1:.1f})')

    ax.plot(x2, y2, 'ro', markersize=8, label=f'Camera ({x2:.1f}, {y2:.1f})')
    ax.text(x2, y2 + 5, f' Camera ({x2:.1f}, {y2:.1f})')

    # ---------------------------------
    # 2-4. Draw gray reference lines
    # ---------------------------------
    ax.plot([x1, x2], [y1, y1], color='gray', linestyle='-', label='Reference Lines')
    ax.plot([x2, x2], [y1, y2], color='gray', linestyle='-')

    # ---------------------------------
    # 2-5. Draw the 27-degree fan
    # ---------------------------------
    fan_radius = length * 1.2
    theta1 = 180 - 13.5  # 166.5 deg
    theta2 = 180 + 13.5  # 193.5 deg

    fan = patches.Wedge(center=(x2, y2), r=fan_radius, theta1=theta1, theta2=theta2,
                        color='gray', alpha=0.3, label='27° Fan (from x- axis)')
    ax.add_patch(fan)

    rad_180 = np.radians(180)
    ax.plot([x2, x2 + fan_radius * np.cos(rad_180)], 
            [y2, y2 + fan_radius * np.sin(rad_180)], 
            'k-', linestyle=':', alpha=0.5, label='x- reference line (180°)')

    # ---------------------------------
    # 2-6. Draw the input vector
    # ---------------------------------
    math_angle_rad = np.radians(180 + angle_input * -1)  # Convert to math angle in radians

    x_end_vec = x2 + length * np.cos(math_angle_rad)
    y_end_vec = y2 + length * np.sin(math_angle_rad)

    ax.plot([x2, x_end_vec], [y2, y_end_vec], 'g-', linewidth=2.5,
            label=f'Input Vector (Length: {length}, Angle: {angle_input}°)')
    
    ax.plot(x_end_vec, y_end_vec, 'gx', markersize=10)
    ax.text(x_end_vec, y_end_vec - 10, f' End ({x_end_vec:.1f}, {y_end_vec:.1f})', color='g')

    # ---------------------------------
    # 2-7. Show legend, Save, and/or Show Plot (수정됨)
    # ---------------------------------
    ax.legend(loc='upper right')
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # dpi=300 for high quality
        print(f"Plot saved to: {save_path}")

    # Show the plot if show_plot is True
    if show_plot:
        plt.show()
    
    # Close the figure to free up memory
    plt.close(fig)

    # ---------------------------------
    # 2-8. Return the end coordinates (추가됨)
    # ---------------------------------
    x_end_int = int(round(x_end_vec))
    y_end_int = int(round(y_end_vec))
    
    return (x_end_int, y_end_int)

# -----------------------------------------------------------------
# 3. Script execution entry point (수정됨)
# -----------------------------------------------------------------
if __name__ == "__main__":

    from enhanced_plot import end_effector_xy

    x1, y1 = end_effector_xy(114,105,51)

    print(x1, y1)

    coords1 = plot_geometry(
        x1=x1, y1=y1, length=220, angle_input=-5,
        show_plot=True,  
        save_path="plot_result_1.png"  # 저장할 파일 이름
    )
    print(f"벡터 끝점 좌표: {coords1}")