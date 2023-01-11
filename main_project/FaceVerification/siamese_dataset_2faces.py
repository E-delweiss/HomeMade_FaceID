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
        dataset_type = 'train'
        if isValSet_bool:
            dataset_type = 'val'

        imgset_self = glob.glob(f'/Users/thierryksstentini/Downloads/dataset/dataset_sven/dataset_moi_sven_cropped/{dataset_type}/*')
        imgset_lfw = glob.glob(f'/Users/thierryksstentini/Downloads/dataset/dataset_sven/dataset_pauline_sven_cropped/{dataset_type}/*')
        self.imgset = imgset_lfw + imgset_self
        
        label_lfw = np.zeros(len(imgset_lfw)).tolist()
        label_self = np.ones(len(imgset_self)).tolist()
        self.labelset = label_lfw + label_self

        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool

    def _preprocess(self, img_PIL:torch.Tensor)->torch.Tensor:
        ### Resizing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((160,160))
        ])
        img_t = transform(img_PIL)

        ### Normalize data
        if self.isNormalize_bool:
            mean, std = (0.3510, 0.3846, 0.4988), (0.2214, 0.2394, 0.2772)
            img_t = torchvision.transforms.Normalize(mean, std)(img_t)

        ### Data augmentation
        if self.isAugment_bool:
            augment = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=[0.6,1], contrast=[1,1.2], saturation=[0,1]),
                torchvision.transforms.RandomAdjustSharpness(12, p=0.5),
                torchvision.transforms.RandomRotation(degrees=(-10,10)),
                torchvision.transforms.RandomPerspective(distortion_scale=0.22, p=0.5, fill=0),
                torchvision.transforms.RandomResizedCrop(size=(100, 100), scale=(0.85, 0.85))
             ])
            img_t = augment(img_t)
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

        return image, torch.tensor(label).to(torch.int32)


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
    dataset = get_validation_dataset(isAugment_bool=False, isNormalize_bool=True)
    print(next(iter(dataset))[1])