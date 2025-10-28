import cv2

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
        print(f"클릭한 위치: x={x}, y={y}")

# 이미지 불러오기
img = cv2.imread("Camera/distance_test1.jpg")

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키로 종료
        break

cv2.destroyAllWindows()

# 이미지 불러오기
img = cv2.imread("Camera/distance_test2.jpg")

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키로 종료
        break

cv2.destroyAllWindows()