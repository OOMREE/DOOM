import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('head.png', 0)  # trainImage
img2 = cv2.imread('Unn.png', 0)  # queryImage

orb = cv2.ORB_create(edgeThreshold=4, patchSize=16)  # ORB

kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

img3 = cv2.drawKeypoints(img1, kp1, None, color=(84, 7, 7))
img4 = cv2.drawKeypoints(img2, kp2, None, color=(192, 0, 0))
cv2.imwrite('tmp_head.png', img3)
cv2.imwrite('tmp_Unn.png', img4)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

img5 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

cv2.imwrite('tmp_input_vs_label.png', img5)

good = matches
for m in matches:
    if m.distance < 0.7:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,# draw only inliers
                   flags=2)

img6 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite('tmp_selection_result.png', img6)

plt.imshow(img6, 'gray')
plt.show()