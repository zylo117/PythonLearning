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
