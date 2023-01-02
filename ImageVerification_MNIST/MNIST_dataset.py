import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np
import PIL

import torch
import torchvision

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, isTrainSet_bool:bool=None, isUnbalanced_bool:bool=False)->tuple:
        self.isTrainSet_bool = isTrainSet_bool
        self.isUnbalanced_bool = isUnbalanced_bool

        self.dataset = torchvision.datasets.MNIST("../dataset", train=self.isTrainSet_bool, download=True, 
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0.1307, 0.3081)
                ]))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx)->tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.dataset[idx]

        ### Unbalance the dataset: keep only the 7 digits, else is 0
        if self.isUnbalanced_bool and self.dataset[idx][1] != 7:
            label = 0

        return img, torch.tensor(label, dtype=torch.int32)



def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using MNIST. 
    """
    dataset = MNISTDataset(isTrainSet_bool=True, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the dataset using MNIST. 
    """
    dataset = MNISTDataset(isTrainSet_bool=False, **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == "__main__":
    from collections import Counter
    dataset = get_validation_dataset(16, isUnbalanced_bool=True)
    labels = next(iter(dataset))[1]
    print(torch.unique(labels))