import os, sys
import glob
import pandas as pd

import numpy as np

import torch
import torchvision


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, val_stride:int=0, isValSet_bool:bool=None, isAugment_bool:bool=False, isNormalize_bool:bool=False)->tuple:
        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool
        self.PN_df = df

        if isValSet_bool:
            assert val_stride > 0, 'val_stride argument must be greater than 0'
            self.PN_df = self.PN_df[::val_stride]
            self.PN_df = self.PN_df.reset_index(drop=True)
        elif val_stride > 0:
            self.PN_df = self.PN_df.drop(self.PN_df.index[::val_stride])
            self.PN_df = self.PN_df.reset_index(drop=True)

    def _preprocess(self, img_t:torch.Tensor)->torch.Tensor:
        ### Resizing
        if type(img_t) is Image.Image:
            img_t = torchvision.transforms.ToTensor()(img_t)
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((160,160))])
        img_t = transform(img_t)

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
        return self.PN_df.shape[0]
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_t = self.PN_df['Images'][idx]
        image = self._preprocess(img_t)

        label = self.PN_df['Labels'][idx]
        
        return image, torch.tensor(label, dtype=torch.int32)


def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = SiameseDataset(root="???", **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, val_stride=???, isValSet_bool=False, isAugment=True, isNormalize=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = SiameseDataset(root="???", **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, val_stride=???, isValSet_bool=True, isAugment=False, isNormalize=True)
    return dataloader