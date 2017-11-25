from collections import deque
import numpy as np
import argparse
import imutils
from imutils.video import WebcamVideoStream
import cv2
from imutils.video import FPS
import datetime
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-t", "--have_trail", type=bool, default=False, help="Whether show trail")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = WebcamVideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

fps = FPS().start()
current_fps = 0

# keep looping
while True:
    if not args.get("video", False):
        frame = camera.read()

    else:
        (grabbed, frame) = camera.read()
        time.sleep(2.0)

    if fps._numFrames % 2 == 0:
        # 帧率计数
        start_time = datetime.datetime.now()

    # if viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=400)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    if fps._numFrames / 4 % 2 == 0:
        blurred_even = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        cv2.threshold(blurred_even, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV, blurred_even)
        # cv2.imshow("BWOld", blurred_even)

    elif fps._numFrames / 4 % 2 == 1:
        blurred_odd = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        cv2.threshold(blurred_odd, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV, blurred_odd)
        # cv2.imshow("BWNew", blurred_odd)

    if fps._numFrames > 4:
        # delta = blurred_odd - blurred_even
        delta = cv2.absdiff(blurred_odd, blurred_even)
        cv2.imshow("Delta", delta)

        delta = cv2.medianBlur(delta, 3)
        cv2.imshow("MedFil", delta)

        delta = cv2.dilate(delta, None, iterations=2)
        cv2.imshow("Dilation", delta)

        delta = cv2.erode(delta, None, iterations=2)
        cv2.imshow("Erosion", delta)
        # cv2.waitKey()
        # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform a series of dilations and erosions to remove any small blobs left in the mask

        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                # calculate the centroid of moment
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # onlt proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame, then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        pts.appendleft(center)

        if args["have_trail"]:
            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and draw the connecting lines
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        if fps._numFrames % 2 == 1:
            # 帧率计数
            end_time = datetime.datetime.now()
            two_frame_time = (end_time - start_time).total_seconds()
            current_fps = (1 / (two_frame_time / 2))
        cv2.putText(frame, str(int(current_fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show the frame to screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

fps.stop()
camera.stop()
cv2.destroyAllWindows()
