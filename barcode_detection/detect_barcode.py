import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
# cv2.imshow("Ori", image)
# cv2.imshow("Gray", gray)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)    # ksize=-1 -> Scharr operator
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# cv2.imshow("gradX", gradX)
# cv2.imshow("gradY", gradY)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
# cv2.imshow("Gradient", gradient)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("Gradient_cvt", gradient)

# blur and threshold the image
blurred = cv2.GaussianBlur(gradient, (19, 19), 0)
# cv2.imshow("GBlurred", blurred)
(_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 711, 0)
cv2.imshow("Thresh", thresh)

# construct a closing kernel and apply it to the thresholded image
# 把所有图像morph成矩形
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 50, h // 50))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Morph", closed)

# perform a series of erosions and dilations, remove small blobs
erode = cv2.erode(closed, None, iterations=4)
cv2.imshow("Erode", erode)
dilate = cv2.dilate(erode, None, iterations=4)
cv2.imshow("Dilate", dilate)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
_, cnts, _ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.array(cv2.boxPoints(rect), dtype=np.int)

# draw a bounding box arounded the detected barcode and display the image
cv2.drawContours(image, [box], -1, (255, 255, 0), 3)
cv2.imshow("Image", image)

cv2.waitKey()
