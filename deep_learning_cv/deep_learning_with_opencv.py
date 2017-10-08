# import the necessary packages
import numpy as np
import argparse
import time  # +1s
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="Path to ImageNet labels (i.e., syn-sets)")
