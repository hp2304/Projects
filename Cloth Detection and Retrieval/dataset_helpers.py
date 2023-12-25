import json
import os
from PIL import Image, ImageDraw
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from data_aug import merge_horizontal


def load_sample(image_id, img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    with open(mask_path) as f:
        meta = json.load(f)

    boxes = []
    masks = []
    labels = []
    for key, val in meta.items():
        if 'item' in key:
            mask = np.zeros((h, w), dtype=bool)
            for polygon in val['segmentation']:
                region = Image.new('L', (w, h), 0)
                ImageDraw.Draw(region).polygon(polygon, outline=1, fill=1)
                region = np.array(region) > 0
                mask = np.logical_or(mask, region)
            boxes.append(val['bounding_box'])
            masks.append(mask)
            # labels.append(val['category_id'])
            labels.append(1)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    image_id = torch.tensor([image_id])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # suppose all instances are not crowd
    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    return img, target


class ClothesDataset(Dataset):
    def __init__(self, data_path, transform=None, is_train=True):
        self.img_path = os.path.join(data_path, "image")
        self.mask_path = os.path.join(data_path, "annos")
        self.filenames = list(sorted(os.listdir(self.img_path)))
        self.dataset_len = len(self.filenames)
        self.transform = transform
        self.colorJitter = transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6)
        self.toTensor = transforms.ToTensor()
        self.is_train = is_train
        with open(os.path.join(data_path, "size_metadata.json"), 'r') as fp:
            self.size_metadata = json.load(fp)

    def __len__(self):
        return self.dataset_len

    def get_labels(self):
        # Cloth type names
        # return ['__background__', 
        #         'short sleeve top',
        #         'long sleeve top',
        #         'short sleeve outwear',
        #         'long sleeve outwear',
        #         'vest',
        #         'sling',
        #         'shorts',
        #         'trousers',
        #         'skirt',
        #         'short sleeve dress',
        #         'long sleeve dress',
        #         'vest dress',
        #         'sling dress']
        return ['__background__',
                'clothing_item']

    def get_height_and_width(self, idx: int):
        dims = self.size_metadata[self.filenames[idx]]
        return dims['height'], dims['width']

    def __getitem__(self, idx):
        fn = self.filenames[idx].split('.')[0]
        img, target = load_sample(idx, os.path.join(self.img_path, self.filenames[idx]),
                                  os.path.join(self.mask_path, fn + '.json'))

        if self.is_train:
            # if np.random.uniform() > 0.5:
            #     random_idx = np.random.randint(self.dataset_len, size=1)[0]
            #     random_img, random_target = load_sample(random_idx, os.path.join(self.img_path, self.filenames[random_idx]),
            #                           os.path.join(self.mask_path, self.filenames[random_idx].split('.')[0] + '.json'))
            #     img, target = merge_horizontal((img, target), (random_img, random_target))

            img = self.colorJitter(img)

        if self.transform is not None:
            img, target = self.transform((img, target))

        '''
        # Check if data aug. is working as it should
        for i in range(target["masks"].size(0)):
            draw = ImageDraw.Draw(img)
            x_min = target["boxes"][i][0]
            y_min = target["boxes"][i][1]
            x_max = target["boxes"][i][2]
            y_max = target["boxes"][i][3]
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width = 5)
            print("Plotting mask ", i)
            plt.imshow(target["masks"][i].numpy())
            plt.show()
            
        print("Plotting image ...")
        plt.imshow(np.array(img))
        plt.show()
        '''

        img = self.toTensor(img)

        return img, target
