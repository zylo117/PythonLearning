# The Canny edge detection algorithm can be broken down into 5 steps:
#
# Step 1: Smooth the image using a Gaussian filter to remove high frequency noise.
# Step 2: Compute the gradient intensity representations of the image.
# Step 3: Apply non-maximum suppression to remove “false” responses to to edge detection.
# Step 4: Apply thresholding using a lower and upper boundary on the gradient values.
# Step 5: Track edges using hysteresis by suppressing weak edges that are not connected to strong edges.

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred", image)

canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
