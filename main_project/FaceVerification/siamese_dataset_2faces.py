import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np
import PIL

import torch
import torchvision


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, isAugment_bool:bool=False, isNormalize_bool:bool=False)->tuple: 
        imgset_self = glob.glob(f'../../dataset/dataset_2faces/dataset_moi_sven_cropped/*')
        imgset_lfw = glob.glob(f'../../dataset/dataset_2faces/dataset_pauline_sven_cropped/*')

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


def get_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = SiameseDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataset = get_validation_dataset(isAugment_bool=False, isNormalize_bool=True)
    print(next(iter(dataset))[1])