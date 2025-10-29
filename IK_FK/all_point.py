import re
import matplotlib.pyplot as plt

points = []

with open("IK_FK/angles_coords_step1.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 빈 줄은 무시
        # 마지막 괄호 안 좌표 추출
        match = re.findall(r"\(([^)]+)\)", line)
        if match:
            last = match[-1]  # 마지막 괄호
            x, y = map(float, last.split(","))
            points.append((x, y))

# x, y 분리
xs, ys = zip(*points)

# 플롯
plt.scatter(xs, ys, color="blue", marker="o")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("마지막 좌표 플롯")
plt.grid(True)
plt.show()
