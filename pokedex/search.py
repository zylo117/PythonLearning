# import the necessnary packages
from descriptor.searcher import Searcher
from descriptor.zernikemoments import ZernikeMoments
import imutils
import numpy as np
import argparse
import cv2
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required=True, help="Path to where the index file will be stored")
ap.add_argument("-q", "--query", required=True, help="Path to the query image")
args = vars(ap.parse_args())

# load the index
index = open(args["index"], "rb").read()
index = pickle.loads(index)

# load the query image, convert it to grayscale, and
# resize it
image = cv2.imread(args["query"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = imutils.resize(image, width=64)

# threshold the image
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
# cv2.imshow("Cnts", thresh)
# cv2.waitKey()

# initialize the outline image, find the outermost
# contours (the outline) of the pokemon, then draw it

outline = np.zeros(image.shape, dtype="uint8")
(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

cv2.drawContours(outline, cnts, 0, 255, -1)
# cv2.imshow("Cnts", outline)
# cv2.waitKey()

# compute Zernike moments to characterize the shape of pokemon outline
desc = ZernikeMoments(21)
queryFeature = desc.describe(outline)

# perform the search to identify the pokemon
searcher = Searcher(index)
results = searcher.search(queryFeature)
print("That pokemon is : %s" % results[0][1].upper())
cv2.imshow("Image", image)
cv2.waitKey()
