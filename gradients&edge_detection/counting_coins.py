import numpy as np
import argparse
import cv2
import auto_canny
import otsu_and_riddler_calvard as orc

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imshow("Image", image)

# 加入二值化
# th_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)     # 自适应二值化
# retval, th_image = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)        # 定值二值化
retval, th_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)      # OTSU最大类间方差阈值化（自动）
# retval, th_image = orc.otsu(blurred)      #mahotas的OTSU算法
# retval, th_image = orc.rc(blurred)      # mahotas的Riddler-Calvard算法
cv2.imshow("Theshold", th_image)

# 抠出硬币图
# new_image = cv2.bitwise_and(image, image, mask=th_image)
# cv2.imshow("Remove", new_image)

edged = auto_canny.auto_canny(th_image)
cv2.imshow("Edged", edged)

# contours: : 检测到的轮廓，每个轮廓存储为一个点向量，即用point类型的vector，例如可为类型vector<vector<Point> >。
# hierarchy: 可选的输出向量，包含图像的拓扑信息。 每个轮廓contours[i],
# hierarchy[i][0] , 后一个轮廓，
# hierarchy[i][1] , 前一个轮廓，
# hierarchy[i][2] , 父轮廓，
# hierarchy[i][3], 内嵌轮廓的索引编号。
# 如果没有对应项，hierarchy[i]中的对应项设为负数。
# mode: 检索模式，可选模式包括
# RETR_EXTERNAL: 只监测最外层轮扩。hierarchy[i][2] ＝ hierarchy[i][3] ＝ －1
# RETR_LIST: 提取所有轮廓，并放置在list中。检测的轮廓不建立等级关系。
# RETR_CCOMP: 提取所有轮廓，并将其组织为双层结构，顶层为联通域的外围边界，次层为空的内层边界。
# RETR_TREE: 提取所有轮廓，并重新建立网状的轮廓结构。
# method: 轮廓的近似办法，包括
# CHAIN_APPROX_NONE: 获取每个轮廓的每个像素，相邻两点像素位置差不超过1，max(abs(x1-x2),abs(y1-y2)) == 1
# CHAIN_APPROX_SIMPLE: 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
# CHAIN_APPROX_TC89_Lland T. Chin. On the detection of dominant points on digital curves. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 11(8):859–872, 1989.
# offSet: 每个轮廓点的可选偏移量，默认Point(), 当ROI图像中找出的轮廓需要在整个图中进行分析时，可利用这个参数。
(_, cnts, _) = cv2.findContours(th_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("I count {} coins in this image".format(len(cnts)))

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (255, 255, 0), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print("Coin #{}".format(i + 1))
    coin = image[y:y + h, x:x + w]

    cv2.imshow("Coin", coin)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask,(int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask=mask))
    cv2.waitKey(0)
