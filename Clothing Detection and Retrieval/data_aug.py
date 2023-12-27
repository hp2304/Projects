from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class Resize(object):
    def __init__(self, desired_width = 512, desired_height = 512):    
        # Make sure parameter values are atlease 224 pixels
        assert(desired_width >= 224 and desired_height >= 224)
        
        self.desired_width = desired_width
        self.desired_height = desired_height
        
    def __call__(self, sample):
        # Get input and its mask from arg.
        img, target = sample[0], sample[1]
        width, height = img.size
        
        w_f = self.desired_width / width
        h_f = self.desired_height / height
        
        # Performs actual resizing of image and mask
        img = img.resize((self.desired_width, self.desired_height), resample = Image.NEAREST)
        
        # Update bounding box coordinates
        target["boxes"][:, 0] = w_f * target["boxes"][:, 0]
        target["boxes"][:, 1] = h_f * target["boxes"][:, 1]
        target["boxes"][:, 2] = w_f * target["boxes"][:, 2]
        target["boxes"][:, 3] = h_f * target["boxes"][:, 3]
        
        # Update their area too
        target["area"] = target["area"] * w_f * h_f
        
        masks = []
        # Also resize masks
        for i in range(target["masks"].size(0)):
            mask_pil = Image.fromarray(target["masks"][i].numpy()*255)
            mask_pil = mask_pil.resize((self.desired_width, self.desired_height), resample = Image.NEAREST)
            masks.append(np.array(mask_pil) > 0)
        
        del target["masks"]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        target["masks"] = masks
        
        return img, target

class HorizontalFlip(object):
    def __init__(self, prob = 0.5):
        assert(prob >= 0 and prob <= 1)
        self.prob = prob
        
    def __call__(self, sample):
        # Get input and its mask from arg.
        img, target = sample[0], sample[1]
        
        # If randomly generated number (b.w. 0 and 1) is > prob then flip (input image and target) horizontally else not.
        if np.random.uniform() < self.prob:
            # Flip the image horizontally
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            width, _ = img.size
            
            # Update bounding box coordinates
            x_mins = target["boxes"][:, 0]
            target["boxes"][:, 0] = width - target["boxes"][:, 2]
            target["boxes"][:, 2] = width - x_mins
            
            # Also flip masks left to right
            # copy = target["masks"].clone().detach()
            masks = target["masks"].numpy()
            target["masks"] = torch.from_numpy(np.flip(masks, axis=2).copy())
        
        return img, target
    
    
class AffineTransform(object):
    def __init__(self, degrees = 0., translate = (0, 0), scale = (1., 1.), shear = (0., 0., 0., 0.), resample = Image.NEAREST):
        assert(degrees >= 0 and degrees <= 180)
        assert(all(val >= 0 and val <= 1 for val in translate))
        assert(all(val > 0 for val in scale))
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        
    def __call__(self, sample):
        # Get image and its mask from arg.
        img, target = sample[0], sample[1]
        width, height = img.size
        
        # Randomly generate angle b.w. [-self.degrees, self.degrees]
        angle = np.random.uniform(-self.degrees, self.degrees)
        
        # Randomly generate dx and dy based on params to translate the image
        max_dx = self.translate[0] * width
        max_dy = self.translate[1] * height
        translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                        np.round(np.random.uniform(-max_dy, max_dy)))
        
        # Randomly generate new scale based on params
        new_scale = np.random.uniform(self.scale[0], self.scale[1])
        
        # Randomly generate shear ranges based on params
        shear_ranges = [np.random.uniform(self.shear[0], self.shear[1]), np.random.uniform(self.shear[2], self.shear[3])]
        
        # Apply affine transform based on above generated values on image and mask
        img = transforms.functional.affine(img, angle = angle, translate = translations,
                                           scale = new_scale, shear = shear_ranges, resample = Image.NEAREST)
        
        boxes = []
        masks = []
        labels = []
        # Also resize masks
        for i in range(target["masks"].size(0)):
            mask_pil = Image.fromarray(target["masks"][i].numpy()*255)
            mask_pil = transforms.functional.affine(mask_pil, angle = angle, translate = translations,
                                           scale = new_scale, shear = shear_ranges, resample = Image.NEAREST)
            np_binary_mask = np.array(mask_pil) > 0
            if np.count_nonzero(np_binary_mask) > 64:
                target["masks"][i] = torch.tensor(np.array(mask_pil) > 0, dtype = torch.uint8)

                # Update bounding box coordinates by getting top left and bottom right points
                np_img = np.array(mask_pil)
                pos = np.where(np_img > 0)
                mins = np.min(pos, 1)
                maxs = np.max(pos, 1)
                
                boxes.append([mins[1], mins[0], maxs[1], maxs[0]])
                masks.append(np_binary_mask)
                labels.append(target["labels"][i].item())
                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
           
        image_id = target["image_id"]
        
        del target
        
        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd 
        
        return img, target

def merge_horizontal(base_sample, random_sample):
    w = 416
    h = 416
    resize = Resize(w, h)
    img1, target1 = resize((base_sample[0], base_sample[1]))
    img2, target2 = resize((random_sample[0], random_sample[1]))
    
    img = Image.new('RGB', (2*w, h))
    img.paste(img1, (0, 0))
    img.paste(img2, (w, 0))
    
    target1["masks"] = torch.cat([target1["masks"], torch.zeros((target1["masks"].size(0), h, w), dtype = torch.uint8)], dim=2)
    target2["masks"] = torch.cat([torch.zeros((target2["masks"].size(0), h, w), dtype = torch.uint8), target2["masks"]], dim=2)
    
    target2["boxes"][:, 0] += w
    target2["boxes"][:, 2] += w
    
    boxes = torch.cat([target1["boxes"], target2["boxes"]], dim=0)
    masks = torch.cat([target1["masks"], target2["masks"]], dim=0)
    labels = torch.cat([target1["labels"], target2["labels"]], dim=0)
    area = torch.cat([target1["area"], target2["area"]], dim=0)
    iscrowd = torch.cat([target1["iscrowd"], target2["iscrowd"]], dim=0)

    image_id = target1["image_id"]

    del img1, target1, img2, target2

    target = {}
    target["boxes"] = boxes
    target["masks"] = masks
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd 

    return img, target