img1 = cv.imread('img1', cv.IMREAD_GRAYSCALE) 
img2 = cv.imread('img2', cv.IMREAD_GRAYSCALE) 
if img1 is None or img2 is None:
    print('Could not open or find the images!') 
    exit(0)
#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian) 
keypoints1, descriptors1 = detector.detectAndCompute(img1, None) # 提取img1特征点
keypoints2, descriptors2 = detector.detectAndCompute(img2, None) # 提取img2特征点
#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED) 
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2) #匹配img1与img2的特征点
#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m) #寻找最佳匹配特征对
#-- Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# 创建包含特征匹配信息的图像
#-- Show detected matches
cv.namedWindow('Good Matches', cv.WINDOW_NORMAL) # 创建图像显示窗口 WINDOW_NORMAL使得图像窗口可以调整
cv.imshow('Good Matches', img_matches) # 显示包含特征匹配信息的图像

points1=np.zeros((len(good_matches),2),dtype=float)
points2=np.zeros((len(good_matches),2),dtype=float)

for i in range(len(good_matches)):
    points1[i,:]=keypoints1[good_matches[i].queryIdx].pt
    points2[i,:]=keypoints2[good_matches[i].trainIdx].pt # 最佳匹配特征点位置

h, mask = cv.findHomography(points1, points2, cv.RANSAC) #根据最佳匹配特征点生成将img1对齐到img2的几何变换矩阵
height, width= img2.shape
img1Reg = cv.warpPerspective(img1, h, (width, height)) # 根据前述生成的几何变换对img1进行变换

#img1Reg即为将img1对齐到img2的新图像