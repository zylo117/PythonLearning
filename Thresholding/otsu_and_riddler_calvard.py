import cv2
import argparse
import mahotas


def otsu(gray_image):
    T = mahotas.otsu(gray_image)
    gray_image[gray_image > T] = 255
    gray_image[gray_image < T] = 0
    gray_image = cv2.bitwise_not(gray_image)
    return T, gray_image


def rc(gray_image):
    T = mahotas.rc(gray_image)
    gray_image[gray_image > T] = 255
    gray_image[gray_image < T] = 0
    gray_image = cv2.bitwise_not(gray_image)
    return T, gray_image

# 测试用代码
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())
#
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("Image", image)
#
# otsu_T = mahotas.otsu(blurred)
# print("Otsu's threshold: {}".format(otsu_T))
#
# otsu_thresh = gray.copy()
# otsu_thresh[otsu_thresh > otsu_T] = 255
# otsu_thresh[otsu_thresh < otsu_T] = 0
# otsu_thresh = cv2.bitwise_not(otsu_thresh)
# cv2.imshow("Otsu", otsu_thresh)
#
# rc_T = mahotas.rc(blurred)
# print("Riddler-Calvard's threshold: {}".format(rc_T))
#
# rc_thresh = gray.copy()
# rc_thresh[rc_thresh > rc_T] = 255
# rc_thresh[rc_thresh < rc_T] = 0
# rc_thresh = cv2.bitwise_not(rc_thresh)
# cv2.imshow("Riddler-Calvard", rc_thresh)
# cv2.waitKey(0)
