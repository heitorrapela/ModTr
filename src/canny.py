import cv2
import numpy as np
import glob
import os

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)   

# ORIGINAL = 'infrared'
# SPLIT = 'test'
# DATASET_PATH = sorted(glob.glob('../datasets/LLVIP/' + ORIGINAL + '/' + SPLIT + '/*.jpg'))
# SAVE_PATH = '../datasets/LLVIP/' + ORIGINAL + '/' + SPLIT + '_edge/'

ORIGINAL = 'JPEGImages'
DATASET = 'FLIR_aligned'
DATASET_PATH = sorted(glob.glob('../datasets/' + DATASET + '/' + ORIGINAL + '/*.jpeg'))
SAVE_PATH = '../datasets/' + DATASET+ '/' + ORIGINAL + '_edge/'


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

for idx, inp_img in enumerate(DATASET_PATH):
    print(idx, " of ", len(DATASET_PATH))
    img = cv2.imread(inp_img)
    edge = CannyDetector()(img, low_threshold=100, high_threshold=200)
    cv2.imwrite(SAVE_PATH + inp_img.split('/')[-1], edge)