import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2

# 构建 argument parser 并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to the JSON configuration file")
args = vars(ap.parse_args())

# 过滤警告，加载配置文件并且初始化Dropbox
# 客户端
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None
