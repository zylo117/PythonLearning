# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FPS
import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
    """
    每只眼有6个点，
        p1  p2
    p0          p3
        p5  p4

    d(a, b)为两点距离
    ear_aspect_ratio = d(p1, p5) / d(p3, p0) + d(p2, p4) / d(p3, p0)

    这样可以保证眼睛两角都闭上的时候才判断为闭眼
    """

    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2. * C)

    # return the eye aspect ratio
    return ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default=0,
                help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# if a video path was not supplied, grab the reference to the webcam
camera = VideoStream(src=args["video"]).start()

fps = FPS().start()
current_fps = 0

# keep looping
while True:
    frame = camera.read()

    if fps._numFrames % 2 == 0:
        # 帧率计数
        start_time = datetime.datetime.now()

    if fps._numFrames % 2 == 1:
        # 帧率计数
        end_time = datetime.datetime.now()
        two_frame_time = (end_time - start_time).total_seconds()
        current_fps = (1 / (two_frame_time / 2))
    cv2.putText(frame, str(int(current_fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # what to do
    # load the input image, resize it, and convert it to grayscale
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart: lEnd]
        rightEye = shape[rStart: rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, leftEyeHull, -1, (255, 0, 255), 1)
        cv2.drawContours(frame, rightEyeHull, -1, (255, 0, 255), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # reset the eye frame counter
            COUNTER = 0

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 出图
    # 播放帧
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 更新FPS计数
    fps.update()

fps.stop()
camera.stop()
cv2.destroyAllWindows()
