# import the necessnary packages
from descriptor.searcher import Searcher
from descriptor.zernikemoments import ZernikeMoments
import imutils
import  numpy as np
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
