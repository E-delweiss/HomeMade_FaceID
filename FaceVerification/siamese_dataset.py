import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np
import PIL

import torch
import torchvision


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, isValSet_bool:bool=None, isAugment_bool:bool=False, isNormalize_bool:bool=False)->tuple:
        imgset_self = glob.glob('../dataset/dataset_moi_mbp_cropped/*')
        imgset_lfw = rd.sample(glob.glob('../dataset/lfw/lfw_funneled/*/*'), len(imgset_self))

        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool

        if isValSet_bool:
            imgset_lfw = rd.sample(imgset_lfw, int(len(imgset_self)*0.3)) 
            imgset_self = rd.sample(imgset_self, int(len(imgset_self)*0.3))
        else :
            imgset_lfw = rd.sample(imgset_lfw, int(len(imgset_self)*0.7)) 
            imgset_self = rd.sample(imgset_self, int(len(imgset_self)*0.7))

        label_lfw = np.ones(len(imgset_lfw)).tolist()
        label_self = np.zeros(len(imgset_self)).tolist()
        self.labelset = label_lfw + label_self
        self.imgset = imgset_lfw + imgset_self

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
        
        ### Normalize data TODO
        if self.isNormalize_bool:
            mean, std = img_t.mean(), img_t.std()
            img_t = (img_t - mean) / std

        return img_t

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imgset[idx]
        img_PIL = PIL.Image.open(img_path).convert('RGB')
        image = self._preprocess(img_PIL)

        label = self.labelset[idx]
        return image, torch.tensor(label, dtype=torch.int32)


def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = SiameseDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = SiameseDataset(**kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


if __name__ == "__main__":
    dataset = get_training_dataset(16, isValSet_bool=False, isAugment_bool=True, isNormalize_bool=False)
    print(next(iter(dataset)))