import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np
import PIL

import torch
import torchvision


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, ratio:int=1, isAugment_bool:bool=False, isNormalize_bool:bool=False)->tuple: 
        self.imgset_self = glob.glob('../../dataset/dataset_moi_mbp_cropped/*')
        self.imgset_lfw = glob.glob('../../dataset/lfw/lfw_funneled/*/*')

        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool
        self.ratio = ratio

        self.label_lfw = np.zeros(len(self.imgset_lfw)).tolist()
        self.label_self = np.ones(len(self.imgset_self)).tolist()
        
        self.labelset = self.label_lfw + self.label_self
        self.imgset = self.imgset_lfw + self.imgset_self

    def _preprocess(self, img_PIL:torch.Tensor)->torch.Tensor:
        ### Resizing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((160,160))
        ])
        img_t = transform(img_PIL)

        ### Data augmentation
        if self.isAugment_bool:
            augment = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=[0.5,1], contrast=[1,1.2], saturation=[0,1]),
                torchvision.transforms.RandomAdjustSharpness(12, p=0.7),
                torchvision.transforms.RandomRotation(degrees=5)
                ])
            img_t = augment(img_t)
        
        ### Normalize data (TBM)
        if self.isNormalize_bool:
            mean, std = (0.4236, 0.3698, 0.3317), (0.2988, 0.2733, 0.2654)
            img_t = torchvision.transforms.Normalize(mean, std)(img_t)
        return img_t

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ### Control the ratio btw true and false faces for the dataloader
        if self.ratio:
            pos_idx = idx // (self.ratio + 1)
            if idx % (self.ratio + 1):
                neg_idx = idx - 1 - pos_idx
                neg_idx = neg_idx % len(self.imgset_lfw)
                img_path = self.imgset_lfw[neg_idx]
                label = self.label_lfw[neg_idx]
            else:
                pos_idx = pos_idx % len(self.imgset_self)
                img_path = self.imgset_self[pos_idx]
                label = self.label_self[pos_idx]
        else:
            img_path = self.imgset[idx]
            label = self.labelset[idx]

        img_PIL = PIL.Image.open(img_path).convert('RGB')
        image = self._preprocess(img_PIL)

        return image, torch.tensor(label, dtype=torch.int32)


def get_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = SiameseDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataset = get_training_dataset(BATCH_SIZE=32, ratio=1, isAugment_bool=True, isNormalize_bool=False)
    print(next(iter(dataset))[1])