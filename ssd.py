import sys
import cv2
import numpy as np

def Image_SSD(src_img, search_img):
    # 1.确定子图的范围
    # 2.遍历子图
    # 3.求模板图和子图的误差平方和
    # 4.返回误差平方和最小的子图左上角坐标
    M = src_img.shape[0]
    m = search_img.shape[0]
    N = src_img.shape[1]
    n = search_img.shape[1]
    Range_x = M - m - 1
    Range_y = N - n - 1
    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    search_img = cv2.cvtColor(search_img, cv2.COLOR_RGB2GRAY)
    min_res = sys.maxsize
    Best_x = 0
    Best_y = 0
    for i in range(Range_x):
        for j in range(Range_y):
            subgraph_img = src_img[i:i+m, j:j+n]
            res = np.sum((search_img.astype("float") - subgraph_img.astype("float")) ** 2) # SSD公式
            if res < min_res:
                min_res = res
                Best_x = i
                Best_y = j
    return Best_x, Best_y

if __name__ == '__main__':
    # 原图路径
    srcImg_path = "C:\\Users\\PC\\Desktop\\SSD\\src.jpg"
    # 搜索图像路径
    searchImg_path = "C:\\Users\\PC\\Desktop\\SSD\\search.jpg"

    src_img = cv2.imread(srcImg_path)
    search_img = cv2.imread(searchImg_path)
    Best_x, Best_y = Image_SSD(src_img, search_img)

    cv2.rectangle(src_img, (Best_y, Best_x), (Best_y + search_img.shape[1], Best_x + search_img.shape[0]), (0, 0, 255), 3)
    cv2.imshow("src_img", src_img)
    cv2.imshow("search_img", search_img)
    cv2.waitKey(0)