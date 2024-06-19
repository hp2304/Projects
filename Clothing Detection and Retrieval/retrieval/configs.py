import os
from typing import Dict

import numpy as np

transform_mat = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]])
ref_vec = np.array([[0.950450, 1.000000, 1.088754]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

LAB_NB_BINS = [10, 3, 3]
NB_ANGLE_BINS = 18

DATASET_DIR_PATH = '/home/hp/Documents/color_detection/deepfashion2/data/train/'
DATASET_IMG_PATH = os.path.join(DATASET_DIR_PATH, 'image')
DATASET_ANNO_PATH = os.path.join(DATASET_DIR_PATH, 'annos')

VAL_DATASET_DIR_PATH = '/home/hp/Documents/color_detection/deepfashion2/data/validation/'
VAL_DATASET_IMG_PATH = os.path.join(VAL_DATASET_DIR_PATH, 'image')
VAL_DATASET_ANNO_PATH = os.path.join(VAL_DATASET_DIR_PATH, 'annos')

SEG_ON = True

# DO NOT EDIT THIS
USER_ORDER = ['hitarth', 'vaibhav', 'saurav', 'parth']
USER_TOTAL_SHARDS: Dict[str, int] = {
    'hitarth': 4,
    'vaibhav': 2,
    'saurav': 14,
    'parth': 14,
}
TOTAL_SHARDS: int = sum(USER_TOTAL_SHARDS.values())

# EDIT THIS
USERNAME = 'hitarth'
USER_SHARDS = list(range(USER_TOTAL_SHARDS[USERNAME]))
