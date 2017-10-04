import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

# 拉普拉斯算法预处理图像
lap = cv2.Laplacian(image, cv2.CV_64F)      # 黑白转换时，要转换到64浮点，避免数据丢失
lap = np.uint8(np.abs(lap))     # 转换为8位无符号整型
cv2.imshow("Laplaction", lap)
cv2.waitKey(0)

# 索贝尔处理，使得可以从X、Y轴计算边缘
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.abs(sobelX))
sobelY = np.uint8(np.abs(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)      # 叠加X/Y的边缘
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)
