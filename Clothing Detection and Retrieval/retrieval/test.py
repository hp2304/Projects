import os
import random
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_feature_store import extract_clothing_items, extract_clothing_item_from_gallery
from configs import DATASET_IMG_PATH, DATASET_ANNO_PATH, VAL_DATASET_IMG_PATH, VAL_DATASET_ANNO_PATH
from extract_features import get_feature_vector_from_img


def test_files(val_files, gallery_feature_store: pd.DataFrame, k: int, gallery_img_path: str, gallery_anno_path: str):
    for fn in val_files:
        img_fp = os.path.join(VAL_DATASET_IMG_PATH, fn)
        anno_fp = os.path.join(VAL_DATASET_ANNO_PATH, fn.replace('.jpg', '.json'))
        clothing_items = extract_clothing_items(img_fp, anno_fp)
        feature_store = gallery_feature_store.iloc[:, 2:].values
        for key, clothing_item in clothing_items.items():
            h, w = clothing_item.shape[:2]
            clothing_item_bw = cv2.cvtColor(clothing_item, cv2.COLOR_RGB2GRAY)
            content_perc = 1 - (np.sum(clothing_item_bw == 0) / (h * w))
            if content_perc < .5:
                continue
            features: np.ndarray = get_feature_vector_from_img(clothing_item)
            features = features.reshape(1, -1)
            numerator = np.abs(feature_store - features)
            denom = np.abs(feature_store + feature_store.mean(1, keepdims=True)) + np.abs(features + features.mean(1))
            dists = (numerator / denom).sum(1).squeeze()
            retrival_indices = np.argpartition(dists, k)[:k]

            _, axs = plt.subplots(4, 4, figsize=(10, 10))
            subplot_idx: int = 0
            axs = axs.ravel()
            for ax in axs:
                ax.set_axis_off()
            axs[subplot_idx].imshow(clothing_item)
            axs[subplot_idx].set_title(f'{fn}: {key}')
            for i in retrival_indices:
                subplot_idx += 1
                gallery_fn = gallery_feature_store.iloc[i, 0]
                gallery_img_item_id = gallery_feature_store.iloc[i, 1]
                gallery_fp = os.path.join(gallery_img_path, gallery_fn)
                gallery_img_anno_fp = os.path.join(gallery_anno_path, gallery_fn.replace('.jpg', '.json'))
                gallery_match = extract_clothing_item_from_gallery(gallery_fp, gallery_img_anno_fp, gallery_img_item_id)

                if subplot_idx % 4 == 0:
                    axs[subplot_idx].imshow(clothing_item)

                    subplot_idx += 1
                axs[subplot_idx].imshow(gallery_match)

                axs[subplot_idx].set_title(f'{gallery_fn}: {gallery_img_item_id}')
            plt.savefig(f'results/{uuid.uuid4().hex.upper()[0:6]}.jpg')
            # plt.show()


def main(feature_df_path: str, nb_samples: int):
    # random.seed(0)
    features_df = pd.read_csv(feature_df_path)
    val_files = random.sample(os.listdir(VAL_DATASET_IMG_PATH), nb_samples)
    test_files(val_files, features_df, 12, DATASET_IMG_PATH, DATASET_ANNO_PATH)


if __name__ == '__main__':
    main('train_seg_on_parth.csv', 500)
