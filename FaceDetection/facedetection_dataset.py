import os, sys
import glob
import pandas as pd

import numpy as np

import torch
import torchvision


class FaceDetectionDataset(Dataset):
    def __init__(self, imgset_face:list, imgset_noface:list, split:str, isValSet_bool:bool=False, 
                 isAugment_bool:bool=False, isNormalize_bool:bool=True):
        """
        Class that build the dataset to feed the Pytorch Dataloader 

        -------------------
        Class attributs:
            imgset_face: list of PIL images
                The list of PIL images with face in it
            imgset_face: list of PIL images.
                The list of PIL images without face in it (typically random
                background found in the 'houseroom dataset').
            split: str
                Used to select training set or validation set
            isValSet_bool: bool
                Boolean to construct a validation dataset
            isAugment_bool: bool
                Boolean to activate the data augmentation preprocessing
            isNormalize_bool: bool
                Boolean to activate normalization of each channel by mean and 
                std ResNet paper values.
        """
        
        self.isAugment_bool = isAugment_bool
        self.isNormalize_bool = isNormalize_bool

        split_pct = float(split.strip('%'))/100
        len_imageset_face = round(len(imgset_face) * split_pct)
        len_imageset_noface = round(len(imgset_noface) * split_pct)

        

        if isValSet_bool == False:
            imgset_face = imgset_face[:len_imageset_face] 
            imgset_noface = imgset_noface[:len_imageset_noface]
        else :
            imgset_face = imgset_face[-len_imageset_face:]
            imgset_noface = imgset_noface[-len_imageset_noface:]
        
        self.imgset = imgset_face + imgset_noface

        label_face = np.ones(len(imgset_face)).tolist()
        label_noface = np.zeros(len(imgset_noface)).tolist()
        self.labelset = label_face + label_noface
        

    def preprocess(self, img)->torch.Tensor:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.ToTensor()])
        img_t = transform(img)
        
        if self.isAugment_bool:
            augment = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop((224,224)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomRotation(degrees=(-10,10))])
            img_t = augment(img_t)

        if self.isNormalize_bool:
            normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
            img_t = normalize(img_t)
        return img_t

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.imgset[idx]
        image_t = self.preprocess(image)
        
        label = self.labelset[idx]
            
        return image_t, torch.tensor(label).to(torch.float32)



def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = FaceDetectionDataset(root="???", **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, val_stride=???, isValSet_bool=False, isAugment=True, isNormalize=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = FaceDetectionDataset(root="???", **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, val_stride=???, isValSet_bool=True, isAugment=False, isNormalize=True)
    return dataloader