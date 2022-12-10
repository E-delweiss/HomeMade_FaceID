import os, sys
import glob
import pandas as pd
import random as rd

import numpy as np

import PIL
import torch
import torchvision


class FaceDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, isValSet_bool:bool=False, isAugment_bool:bool=False, isNormalize_bool:bool=True):
        """
        Class that build the dataset to feed the Pytorch Dataloader 

        -------------------
        Class attributs:
            TODO
            isValSet_bool: bool
                Boolean to construct a validation dataset
            isAugment_bool: bool
                Boolean to activate the data augmentation preprocessing
            isNormalize_bool: bool
                Boolean to activate normalization of each channel by mean and 
                std ResNet paper values.
        """
        imgset_face_path = rd.sample(glob.glob('../dataset/lfw/lfw_funneled/*/*'), 3000)
        imgset_background_path = rd.sample(glob.glob('../dataset/House_Room/*/*'), 3000)

        self.isAugment_bool = isAugment_bool
        self.isNormalize_bool = isNormalize_bool

        if isValSet_bool:
            imgset_face_path = rd.sample(imgset_face_path, int(3000*0.3)) 
            imgset_background_path = rd.sample(imgset_background_path, int(3000*0.3))
        else :
            imgset_face_path = rd.sample(imgset_face_path, int(3000*0.7)) 
            imgset_background_path = rd.sample(imgset_background_path, int(3000*0.7))
        
        self.imgset = imgset_face_path + imgset_background_path

        label_face = np.ones(len(imgset_face_path)).tolist()
        label_background = np.zeros(len(imgset_background_path)).tolist()
        self.labelset = label_face + label_background


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
        image_PIL = PIL.Image.open(image).convert('RGB')
        image_t = self.preprocess(image_PIL)
        
        label = self.labelset[idx]
            
        return image_t, torch.tensor(label).to(torch.float32)



def get_training_dataset(BATCH_SIZE=16):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = FaceDetectionDataset(isValSet_bool=False, isAugment_bool=True, isNormalize_bool=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = FaceDetectionDataset(isValSet_bool=True, isAugment_bool=False, isNormalize_bool=True)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataset = get_validation_dataset(16)
    # print(next(iter(dataset)))
    