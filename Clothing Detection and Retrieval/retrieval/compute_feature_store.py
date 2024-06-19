import json
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageDraw
from PIL import Image

from configs import DATASET_IMG_PATH, DATASET_ANNO_PATH, TOTAL_SHARDS, USER_ORDER, USER_TOTAL_SHARDS, USERNAME, SEG_ON
from extract_features import get_feature_vector_from_img


def get_user_shard(nb_files: int) -> Tuple[int, int]:
    shard_len = int(round(nb_files / TOTAL_SHARDS))
    start_shard_idx = 0
    for username in USER_ORDER:
        if username == USERNAME:
            break
        start_shard_idx += USER_TOTAL_SHARDS[username]
    start_idx = start_shard_idx * shard_len
    end_idx = start_idx + (shard_len * USER_TOTAL_SHARDS[USERNAME])
    end_idx = end_idx if end_idx < nb_files else nb_files
    return start_idx, end_idx


def get_mask_from_polygons(polygons, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for polygon in polygons:
        region = Image.new('L', (w, h), 0)
        ImageDraw.Draw(region).polygon(polygon, outline=1, fill=1)
        region = np.array(region) > 0
        mask = np.logical_or(mask, region)
    return mask


def extract_clothing_item_from_gallery(img_fp: str, anno_fp: str, item_id: str):
    img = cv2.imread(img_fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    with open(anno_fp) as f:
        meta = json.load(f)
    for key, val in meta.items():
        if item_id == key:
            masked_img = img
            if SEG_ON:
                mask = get_mask_from_polygons(val['segmentation'], w, h)
                masked_img[~mask, :] = [0, 0, 0]
            bbox = val['bounding_box']
            return masked_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def extract_clothing_items(img_fp: str, anno_fp: str) -> Dict[str, np.ndarray]:
    img = cv2.imread(img_fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    with open(anno_fp) as f:
        meta = json.load(f)
    clothing_items = {}
    for key, val in meta.items():
        if 'item' in key:
            masked_img = img.copy()
            if SEG_ON:
                mask = get_mask_from_polygons(val['segmentation'], w, h)
                masked_img[~mask, :] = [0, 0, 0]
            bbox = val['bounding_box']
            clothing_items[key] = masked_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return clothing_items


def process(filename: str) -> List:
    img_fp = os.path.join(DATASET_IMG_PATH, filename)
    anno_fp = os.path.join(DATASET_ANNO_PATH, filename.replace('.jpg', '.json'))
    clothing_items_features = []
    clothing_items = extract_clothing_items(img_fp, anno_fp)
    for key, clothing_item in clothing_items.items():
        clothing_item_features = [filename]
        features: np.ndarray = get_feature_vector_from_img(clothing_item)
        clothing_item_features.append(key)
        clothing_item_features.extend(features.tolist())
        clothing_items_features.append(clothing_item_features)
    # print(clothing_items_features)
    return clothing_items_features


def main():
    warnings.filterwarnings("ignore")
    img_filenames = os.listdir(DATASET_IMG_PATH)
    img_filenames.sort()
    start_idx, end_idx = get_user_shard(len(img_filenames))
    with Pool(cpu_count()) as p:
        results = p.map(process, img_filenames[start_idx:end_idx])
        rows = []
        for img_result in results:
            rows.extend(img_result)
        cols = ['filename', 'itemID']
        cols.extend([f'feature_{i + 1}' for i in range(108)])
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(f'train_seg_{"on" if SEG_ON else "off"}_{USERNAME}.csv', index=False)


if __name__ == '__main__':
    main()
