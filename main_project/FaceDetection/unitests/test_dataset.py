import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from facedetection_dataset import get_training_dataset, get_validation_dataset

class TestFaceDataset(unittest.TestCase):
    def __init__(self, TestFaceDataset) -> None:
        super().__init__(TestFaceDataset)
        self.SIZE = 224
        self.BATCH_SIZE = 32
        
    def test_facedataset_trainset(self): 
        train_set = get_training_dataset(self.BATCH_SIZE)
        one_batch = next(iter(train_set))
        idx = np.random.randint(self.BATCH_SIZE)
        img = one_batch[0][idx]
        label = one_batch[1][idx]

        self.assertIs(type(one_batch), list)
        self.assertIs(type(img), torch.Tensor)
        self.assertIs(type(label), torch.Tensor)
        self.assertIs(len(one_batch[0]), self.BATCH_SIZE)
        self.assertIs(len(one_batch[1]), self.BATCH_SIZE)

    def test_facedataset_valset(self): 
        train_set = get_validation_dataset(self.BATCH_SIZE)
        one_batch = next(iter(train_set))
        idx = np.random.randint(self.BATCH_SIZE)
        img = one_batch[0][idx]
        label = one_batch[1][idx]

        self.assertIs(type(one_batch), list)
        self.assertIs(type(img), torch.Tensor)
        self.assertIs(type(label), torch.Tensor)
        self.assertIs(len(one_batch[0]), self.BATCH_SIZE)
        self.assertIs(len(one_batch[1]), self.BATCH_SIZE)
        
if __name__ == "__main__":
    unittest.main()
