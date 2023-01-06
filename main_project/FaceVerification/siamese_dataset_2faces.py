import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np
import PIL

import torch
import torchvision


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, ratio:int, isValSet_bool:bool=None, isAugment_bool:bool=False, isNormalize_bool:bool=False)->tuple: 
        # imgset_lfw = rd.sample(glob.glob('../dataset/lfw/lfw_funneled/*/*'), len(imgset_self))

        imgset_self = glob.glob('/Users/thierryksstentini/Downloads/dataset/dataset_sven/dataset_moi_sven_cropped/*')
        imgset_lfw = glob.glob('/Users/thierryksstentini/Downloads/dataset/dataset_sven/dataset_pauline_sven_cropped/*')
        len_self = len(imgset_self)
        len_lfw = len(imgset_lfw)

        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool
        
        label_lfw = np.zeros(len(imgset_lfw)).tolist()
        label_self = np.ones(len(imgset_self)).tolist()
        
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
                torchvision.transforms.RandomRotation(degrees=(-10,10)),
                torchvision.transforms.RandomPerspective(distortion_scale=0.22, p=0.5, fill=0),
                torchvision.transforms.RandomResizedCrop(size=(100, 100), scale=(0.85, 0.85))
             ])
            img_t = augment(img_t)
        
        ### Normalize data
        if self.isNormalize_bool:
            mean, std = (0.3533, 0.3867, 0.5007), (0.2228, 0.2410, 0.2774)
            img_t = torchvision.transforms.Normalize(mean, std)(img_t)
        return img_t

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imgset[idx]
        label = self.labelset[idx]

        img_PIL = PIL.Image.open(img_path).convert('RGB')
        image = self._preprocess(img_PIL)

        return image, torch.tensor(label, dtype=torch.int32)


def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = SiameseDataset(isValSet_bool=False, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = SiameseDataset(isValSet_bool=True, **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataset = get_training_dataset(ratio=1, isAugment_bool=True, isNormalize_bool=False)
    print(next(iter(dataset))[1])