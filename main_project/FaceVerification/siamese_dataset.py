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
        if isValSet_bool:
            self.imgset_self = glob.glob('../../dataset/new_img_MBA_2023/val_set/moi/*')
            self.imgset_noself = glob.glob('../../dataset/new_img_MBA_2023/val_set/pasmoi/*')
        else :
            self.imgset_self = glob.glob('../../dataset/new_img_MBA_2023/train_set/moi/*')
            self.imgset_noself = glob.glob('../../dataset/new_img_MBA_2023/train_set/pasmoi/*')
        
        
        len_self = len(self.imgset_self)
        len_noself = len(self.imgset_noself)

        self.isNormalize_bool = isNormalize_bool
        self.isAugment_bool = isAugment_bool
        self.isValSet_bool = isValSet_bool
        self.ratio = ratio

        self.label_self = np.ones(len_self).tolist()
        self.label_noself = np.zeros(len_noself).tolist()
        
        self.labelset = self.label_noself + self.label_self
        self.imgset = self.imgset_noself + self.imgset_self


    def _preprocess(self, img_PIL:torch.Tensor)->torch.Tensor:
        ### Resizing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((160,160))
        ])
        img_t = transform(img_PIL)

        ### Normalize data
        if self.isNormalize_bool:
            mean, std = (0.5494, 0.4396, 0.4001), (0.2402, 0.1993, 0.2171)
            img_t = torchvision.transforms.Normalize(mean, std)(img_t)

        ### Data augmentation
        if self.isAugment_bool:
            augment = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=[0.5,1], contrast=[1,1.2], saturation=[0,1]),
                torchvision.transforms.RandomAdjustSharpness(12, p=0.7),
                torchvision.transforms.RandomRotation(degrees=5)
                ])
            img_t = augment(img_t)
        
        return img_t

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if self.isValSet_bool:
            # torch.manual_seed(0)

        if self.ratio:
            pos_idx = idx // (self.ratio + 1)

            if idx % (self.ratio + 1):
                neg_idx = idx - 1 - pos_idx
                neg_idx = neg_idx % len(self.imgset_noself)
                img_path = self.imgset_noself[neg_idx]
                label = self.label_noself[neg_idx]
            else:
                pos_idx = pos_idx % len(self.imgset_self)
                img_path = self.imgset_self[pos_idx]
                label = self.label_self[pos_idx]
        else:
            if self.isValSet_bool:
                rd.seed(idx)
            img_path = self.imgset[idx]
            label = self.labelset[idx]
            # idx_pos = rd.randint(0, len(self.imgset_self)-1)
            # idx_neg = rd.randint(0, len(self.imgset_lfw)-1)
            # img_path_pos = self.imgset_self[idx_pos]
            # img_path_neg = self.imgset_lfw[idx_neg]
            # label_pos = self.label_self[idx_pos]
            # label_neg = self.label_lfw[idx_neg]


        img_PIL = PIL.Image.open(img_path).convert('RGB')
        image = self._preprocess(img_PIL)

        # img_PIL_pos = PIL.Image.open(img_path_pos).convert('RGB')
        # img_PIL_neg = PIL.Image.open(img_path_neg).convert('RGB')
        # image_pos = self._preprocess(img_PIL_pos)
        # image_neg = self._preprocess(img_PIL_neg)

        return image, torch.tensor(label).to(torch.int32)
        # return image_pos, image_neg, torch.tensor(label_pos).to(torch.int32), torch.tensor(label_neg).to(torch.int32)


def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = SiameseDataset(isValSet_bool=False, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=0, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = SiameseDataset(isValSet_bool=True, **kwargs)
    if not BATCH_SIZE:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


if __name__ == "__main__":
    dataset = get_validation_dataset(ratio=0, isAugment_bool=True, isNormalize_bool=False)
    print(next(iter(dataset))[1].shape)