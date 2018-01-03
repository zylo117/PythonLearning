# import the necessary packages
from imutils import paths
import numpy as np
import cv2

# load the class labels from disk
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

# grab the paths to the input images
imagePaths = sorted(list(paths.list_images("images/")))
# (1) load the first image from disk
# (2) pre-process it by resizing it to 224x224 pixels
# (3) construct a blob that can be passed through the pre-trained network
image = cv2.imread(imagePaths[0])
resized = cv2.resize(image, (224, 224))
r_mean = np.mean(resized[:, :, 0])
g_mean = np.mean(resized[:, :, 1])
b_mean = np.mean(resized[:, :, 2])
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (r_mean, g_mean, b_mean))
print("First Blob: {}".format(blob.shape))

print()
