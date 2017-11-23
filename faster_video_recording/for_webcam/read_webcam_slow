from imutils.video import webcamvideostream
from imutils.video import FPS
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=200, help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# slow version (no threading)
# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling frames from webcam...")
stream = cv2.VideoCapture(0)
fps = FPS().start()

# loop over some frames
while fps._numFrames < args["num_frames"]:
    # grab the frames from the stream and resize it to have a maximum width of 400 pixels
    (grabbed, frame) = stream.read()
    frame = imutils.resize(frame, width=400)

    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

stream.release()
cv2.destroyAllWindows()
