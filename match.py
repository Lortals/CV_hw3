import numpy as np
import cv2
import math
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

img_c1=cv2.imread('left.jpg',0)
img_c1_CC=cv2.imread('left.jpg',1)
img_c1_CC=cv2.cvtColor(img_c1_CC,cv2.COLOR_BGR2RGB)
Harris_c1 = cv2.cornerHarris(img_c1, 3, 3, 0.04)
dst_c1 = cv2.dilate(Harris_c1, None)  #将可以标出来的点粗化
plt.figure(figsize=(20, 20))
img_c1_C=img_c1_CC.copy()
thres = 0.1*dst_c1.max()
img_c1_C[dst_c1 > thres] = [255,0,0]
plt.subplot(121)
plt.imshow(img_c1_C)
plt.title('c1')
plt.axis('off')

img_c2=cv2.imread('right.jpg',0)
img_c2_CC=cv2.imread('right.jpg',1)
img_c2= cv2.resize(img_c2, (int(img_c1.shape[1]), int(img_c1.shape[0])))
img_c2_CC = cv2.resize(img_c2_CC, (int(img_c1_CC.shape[1]), int(img_c1_CC.shape[0])))
img_c2_CC=cv2.cvtColor(img_c2_CC,cv2.COLOR_BGR2RGB)
# img_house_C_fuction=img_house_C.copy()
Harris_c2 = cv2.cornerHarris(img_c2, 3, 3, 0.04)
dst_c2 = cv2.dilate(Harris_c2, None)  #将可以标出来的点粗化
img_c2_C=img_c2_CC.copy()
thres = 0.05*dst_c2.max()
img_c2_C[dst_c2 > thres] = [255,0,0]
plt.subplot(122)
plt.imshow(img_c2_C)
plt.title('c2')
plt.axis('off')

# 从一幅Harris响应图像中返回角点，min_dist为分割角点和图像边界的最少像素数目
def get_harris_points(harrisim,min_dist=10,threshold=0.1):   
    # 寻找高于阈值的候选角点
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1  
    # 得到候选点的坐标
    coords = array(harrisim_t.nonzero()).T  
    # 以及它们的 Harris 响应值
    candidate_values = [harrisim[c[0],c[1]] for c in coords]   
    # 对候选点按照 Harris 响应值进行排序
    index = argsort(candidate_values)[::-1] 
    # 将可行点的位置保存到数组中
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1    
    # 按照 min_distance 原则，选择最佳 Harris 点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), 
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0    
    return filtered_coords

#对于每个返回的点，返回点周围2*wid+1个像素的值（假设选取点的min_distance > wid）
def get_descriptors(image, filtered_coords, wid=5):
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - wid:coords[0] + wid + 1,
                coords[1] - wid:coords[1] + wid + 1].flatten()
        desc.append(patch)
    return desc

#对于第一幅图像中的每个角点描述子，使用归一化互相关，选取它在第二幅图像中的匹配角点
def match(desc1, desc2, threshold=0.5):
    n = len(desc1[0])
    # 点对的距离
    d = -ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i, j] = ncc_value
    ndx = argsort(-d)   #从大0到小排序
    matchscores = ndx[:, 0]   #最大一个数的位置坐标
    return matchscores

#两边对称版本的match（）
def match_twosided(desc1, desc2, threshold=0.5):
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)
    ndx_12 = where(matches_12 >= 0)[0]
    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12

#返回将两幅图像并排拼接成的一幅新图像
def appendimages(im1, im2):
    row1 = im1.shape[0]
    row2 = im2.shape[0]
    if row1 < row2:
        im1 = concatenate((im1, zeros((row2 - row1, im1.shape[1]))), axis=0)
    elif row1 > row2:
        im2 = concatenate((im2, zeros((row1 - row2, im2.shape[1]))), axis=0)
    return concatenate((im1, im2), axis=1)
                                    
#显示一幅带有连接匹配之间连线的图片
#输入：im1，im2（数组图像），locs1，locs2（特征位置），matchscores（match的输出），
def plot_matches(im1, im2, locs1, locs2, matchscores):
    im3 = appendimages(im1, im2)
    imshow(im3)
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
    axis('off')

from pylab import *
from numpy import *
wid=9   #比较像素点数目
filtered_coords1 = get_harris_points(dst_c1, wid+1,0.1)   #图1大于阈值的坐标
filtered_coords2 = get_harris_points(dst_c2, wid+1,0.1)   #图2大于阈值的坐标
d1 = get_descriptors(img_c1_CC, filtered_coords1, wid)
d2 = get_descriptors(img_c2_CC, filtered_coords2, wid)
matches = match_twosided(d1, d2,0.8)                 #图1的阈值点与图二哪个阈值点相关度最高，输出与图一相关性最大点的坐标
plt.figure(figsize=(30, 20))
plot_matches(img_c1_CC, img_c2_CC,filtered_coords1, filtered_coords2, matches)
plt.show()