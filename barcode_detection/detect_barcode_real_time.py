import argparse
import imutils
from imutils.video.videostream import VideoStream
import cv2
from imutils.video import FPS
import datetime
import numpy as np

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # cv2.imshow("Ori", image)
    # cv2.imshow("Gray", gray)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # ksize=-1 -> Scharr operator
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # cv2.imshow("gradX", gradX)
    # cv2.imshow("gradY", gradY)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    # cv2.imshow("Gradient", gradient)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow("Gradient_cvt", gradient)
    # blur and threshold the image
    blurred = cv2.GaussianBlur(gradient, (19, 19), 0)
    # cv2.imshow("GBlurred", blurred)
    blurred[blurred < 150] = 0
    cv2.imshow("Casted", blurred)
    (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 711, 0)
    cv2.imshow("Thresh", thresh)
    # construct a closing kernel and apply it to the thresholded image
    # 把所有图像morph成矩形
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 40, h // 70))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Morph", closed)
    # perform a series of erosions and dilations, remove small blobs
    erode = cv2.erode(closed, None, iterations=4)
    # cv2.imshow("Erode", erode)
    dilate = cv2.dilate(erode, None, iterations=4)
    # cv2.imshow("Dilate", dilate)
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    _, cnts, _ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) != 0:
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.array(cv2.boxPoints(rect), dtype=np.int)

        return box
    else:
        return None

# if a video path was not supplied, grab the reference to the webcam
camera = VideoStream(src=0).start()

fps = FPS().start()
current_fps = 0

# keep looping
while True:
    frame = camera.read()

    if fps._numFrames % 2 == 0:
        # 帧率计数
        start_time = datetime.datetime.now()

    frame = imutils.resize(frame, width=400)

    # draw box
    box = detect(frame)
    # draw a bounding box arounded the detected barcode and display the image
    if box is not None:
        cv2.drawContours(frame, [box], -1, (255, 255, 0), 3)

    if fps._numFrames % 2 == 1:
        # 帧率计数
        end_time = datetime.datetime.now()
        two_frame_time = (end_time - start_time).total_seconds()
        current_fps = (1 / (two_frame_time / 2))
    cv2.putText(frame, str(int(current_fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 出图
    # 播放插入帧
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 更新FPS计数
    fps.update()

fps.stop()
camera.stop()
cv2.destroyAllWindows()

