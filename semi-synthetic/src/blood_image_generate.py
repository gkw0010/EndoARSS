import cv2
import numpy as np
from image_generator import BloodDroplet
import os


# 读取输入图像

image_root = '/mnt/data1_hdd/wgk/libmtllast/datasets/DS/test/3/image/'
images = os.listdir(image_root)
for image in images:
    image_path = image_root + image
    original_image = cv2.imread(image_path)
    # 获取原始图像的高度和宽度
    height, width = original_image.shape[:2]

    # 在图像中随机选择一个位置生成血滴
    result_image = np.zeros(original_image.shape)
    for i in range(36):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(10, 32)
        # 根据给定的圆心、半径和图像大小创建血滴实例
        blood_droplet = BloodDroplet.from_circle(cx, cy, radius, height, width)

        result_image += blood_droplet.frame

    mask = np.ones(result_image.shape)
    for m in range(height):
        for j in range(width):
            if result_image[m][j][0] != 0:
                mask[m][j] = [0,0,0]
    original_image = original_image*mask
    original_image += result_image

    output_path = '/mnt/data1_hdd/wgk/libmtllast/datasets/DS_robust/class3/bleeding/5/'+image
    print(output_path)
    cv2.imwrite(output_path, original_image)
    # 保存输入图像和输出图像到指定路径
