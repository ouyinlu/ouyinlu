from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size


        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        '''
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size,ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        
        
        # img, mask = crop(img, mask, self.size,ignore_value)
        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        mask_s1,mask_s2 = deepcopy(mask), deepcopy(mask)

        
        # 随机数据增强分支
        
        img_s1,mask_s1 = ima_aug_geometric_transformation(img_s1, mask_s1)
        img_s1 = agumentation(img_s1,2)

        # 强数据增强分支
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)
        
        ignore_mask_s1 = Image.fromarray(np.zeros((mask_s1.size[1], mask_s1.size[0])))
        ignore_mask_s2 = Image.fromarray(np.zeros((mask_s2.size[1], mask_s2.size[0])))

        img_s1, ignore_mask_s1 = normalize(img_s1, ignore_mask_s1)
        img_s2, ignore_mask_s2 = normalize(img_s2, ignore_mask_s2)

        mask_s1 = torch.from_numpy(np.array(mask_s1)).long()
        mask_s2 = torch.from_numpy(np.array(mask_s2)).long()
        ignore_mask_s1[mask_s1 == 254] = 255
        ignore_mask_s2[mask_s2 == 254] = 255

        return normalize(img_w), img_s1, img_s2,ignore_mask_s1,ignore_mask_s2,cutmix_box2
    
        '''
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        
        
        if self.mode == 'train_l':
            return normalize(img, mask)
        
        img,mask = ima_aug_geometric_transformation(img, mask)
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        img_s1 = agumentation(img_s1,3,True,False)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        img_s2 = agumentation(img_s2,3,True,True)
        # img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
        
    def __len__(self):
        return len(self.ids)
