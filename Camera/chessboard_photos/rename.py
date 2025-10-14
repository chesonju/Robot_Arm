import os

# 현재 작업 디렉토리에서 실행
folder_path = os.getcwd()
jpg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

for idx, filename in enumerate(jpg_files, start=1):
    new_name = f"{idx:04}.jpg"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    
    # 이름이 이미 같은 경우는 건너뜀
    if filename != new_name:
        os.rename(src, dst)
        print(f"{filename} -> {new_name}")
