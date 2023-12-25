import argparse
import json
import os
import random
from typing import List
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np


def get_random_color() -> List[int]:
    return [val for val in random.sample(range(256), 3)]


def get_random_colors(n: int) -> List[List[int]]:
    return [get_random_color() for _ in range(n)]


def get_gt(mask_path: str, img_height: int, img_width: int):
    with open(mask_path) as f:
        meta = json.load(f)

    boxes = []
    masks = []

    for key, val in meta.items():
        if 'item' in key:
            mask = np.zeros((img_height, img_width), dtype=bool)
            for polygon in val['segmentation']:
                region = Image.new('L', (img_width, img_height), 0)
                ImageDraw.Draw(region).polygon(polygon, outline=1, fill=1)
                region = np.array(region) > 0
                mask = np.logical_or(mask, region)
            boxes.append(val['bounding_box'])
            masks.append(mask)

    return boxes, masks


def draw_boxes(img, boxes, colors) -> np.ndarray:
    if boxes is not None:
        for i, box in enumerate(boxes):
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colors[i], 2)

    return img


def draw_masks(img: np.ndarray, masks, colors, alpha: float = 0.5) -> np.ndarray:
    if masks is not None:
        for i, mask in enumerate(masks):
            mask_uint8 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            mask_uint8[mask] = 255

            color_mask = np.zeros_like(img)
            color_mask[mask] = colors[i]

            contours, _ = cv2.findContours(mask_uint8, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=colors[i], thickness=3,
                             lineType=cv2.LINE_AA)

            img = cv2.addWeighted(color_mask, alpha, img, 1, 0)

    return img


def draw_annotations(img: np.ndarray, boxes=None, masks=None) -> np.ndarray:
    colors = get_random_colors(len(boxes))
    img_copy = deepcopy(img)

    img_copy = draw_boxes(img_copy, boxes, colors)
    img_copy = draw_masks(img_copy, masks, colors)

    return img_copy


def main(args: argparse.Namespace) -> None:
    img_dir_path = os.path.join(args.data_dir_path, 'image')
    mask_dir_path = os.path.join(args.data_dir_path, 'annos')

    img_fns = os.listdir(img_dir_path)
    for img_fn in random.sample(img_fns, args.nb_samples):
        img = cv2.imread(os.path.join(img_dir_path, img_fn))
        cv2.imwrite(os.path.join(args.out_dir_path, img_fn), img)
        h, w = img.shape[:2]

        fn = img_fn.split('.')[0]
        boxes, masks = get_gt(os.path.join(mask_dir_path, fn + '.json'), h, w)

        img = draw_annotations(img, boxes, masks)

        out_fn = img_fn.split('.')[0] + '_anno.jpg'
        cv2.imwrite(os.path.join(args.out_dir_path, out_fn), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-i',
                        '--data_dir_path',
                        type=str,
                        required=True,
                        help='Directory path of the dataset')
    parser.add_argument('-n',
                        '--nb_samples',
                        type=int,
                        default=4,
                        help='Number of random samples to draw')
    parser.add_argument('-o',
                        '--out_dir_path',
                        type=str,
                        required=True,
                        help='The end results will be stored inside this directory')

    args = parser.parse_args()
    if not os.path.isdir(args.out_dir_path):
        os.makedirs(args.out_dir_path)
    main(args)

# [How to Run] e.g. python draw_samples.py -i data/train/ -n 20 -o drawings
