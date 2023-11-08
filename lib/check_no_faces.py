# 检测空文件夹
import os
from glob import glob


def find_empty_folders(directorys):
    empty_folders = []
    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if not os.listdir(folder_path):  # 检查文件夹是否为空
                    empty_folders.append(folder_path)

    return empty_folders


def check_png_name(directorys):
    hard_imgs = []
    for directory in directorys:
        img_paths = glob(os.path.join(directory, "**", "*.png"))
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            if img_name != "frame_0.png":
                print(img_path)


# 指定要检测的目录路径
directory_path = [
    "/home/hcr/code/CADDM/test_images6/original_sequences/youtube/c40/frames",
    "/home/hcr/code/CADDM/test_images6/manipulated_sequences/Deepfakes/c40/frames",
    "/home/hcr/code/CADDM/test_images6/manipulated_sequences/Face2Face/c40/frames",
    "/home/hcr/code/CADDM/test_images6/manipulated_sequences/FaceShifter/c40/frames",
    "/home/hcr/code/CADDM/test_images6/manipulated_sequences/FaceSwap/c40/frames",
    "/home/hcr/code/CADDM/test_images6/manipulated_sequences/NeuralTextures/c40/frames",
]

check_png_name(directory_path)

# empty_folders_list = find_empty_folders(directory_path)
#
# # 输出空文件夹的路径列表
# for folder in empty_folders_list:
#     print(folder)
