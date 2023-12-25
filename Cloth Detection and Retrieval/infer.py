import argparse

import os
import random
from copy import deepcopy
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models.detection.mask_rcnn import (MaskRCNN,
                                                    resnet_fpn_backbone)
import torchvision.transforms as T

def load_model(model_ckpt_path: str) -> nn.Module:
    backbone = resnet_fpn_backbone('resnet18', pretrained=True, trainable_layers=4)
    model = MaskRCNN(backbone, num_classes=2)

    checkpoint = torch.load(model_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model


def get_random_color() -> List[int]:
    return [val for val in random.sample(range(256), 3)]


def get_random_colors(n: int) -> List[List[int]]:
    return [get_random_color() for _ in range(n)]


@torch.no_grad()
def predict(fp: str, model: nn.Module, conf_thresh: float = 0.5, mask_thresh: float = 0.5) -> Tuple[List[int], List[np.ndarray]]:
    img = Image.open(fp).convert("RGB")

    test_transform = T.Compose([T.ToTensor()])
    img_transformed = test_transform(img)

    preds = model([img_transformed])
    boxes = []
    masks = []

    if len(preds) > 0:
        preds = preds[0]

        preds_np = {}
        for k in ['boxes', 'masks', 'scores']:
            preds_np[k] = preds[k].cpu().detach().numpy()
        for idx in range(preds_np['scores'].shape[0]):
            if preds_np['scores'][idx] > conf_thresh:
                box = preds_np['boxes'][idx].tolist()
                box = list(map(lambda v: int(v), box))
                mask = preds_np['masks'][idx][0] > mask_thresh

                boxes.append(box)
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
    model = load_model(args.model_ckpt_path)
    model.eval()

    img_fns = os.listdir(args.img_dir_path)
    for img_fn in random.sample(img_fns, args.nb_samples):
        img = cv2.imread(os.path.join(args.img_dir_path, img_fn))
        cv2.imwrite(os.path.join(args.out_dir_path, img_fn), img)

        boxes, masks = predict(os.path.join(args.img_dir_path, img_fn), model, args.conf_thresh)

        img = draw_annotations(img, boxes, masks)

        out_fn = img_fn.split('.')[0] + '_pred.jpg'
        cv2.imwrite(os.path.join(args.out_dir_path, out_fn), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-i',
                        '--img_dir_path',
                        type=str,
                        required=True,
                        help='Directory containing test images')
    parser.add_argument('-m',
                        '--model_ckpt_path',
                        type=str,
                        required=True,
                        help='PyTorch model checkpoint filepath (such has .pth or .pt extension)')
    parser.add_argument('-t',
                        '--conf_thresh',
                        type=float,
                        default=0.7,
                        help='Confidence threshold to filter predictions')
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

# [How to Run] e.g. python infer.py -i data/test/image/ -m models_iter2/model_5.pth -n 10 -o temp
