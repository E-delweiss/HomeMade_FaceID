import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from MNIST_dataset import get_training_dataset, get_validation_dataset

class TestMNISTDataset(unittest.TestCase):
    def __init__(self, TestMNISTDataset) -> None:
        super().__init__(TestMNISTDataset)
        self.SIZE = 28
        self.BATCH_SIZE = 64
        
    def test_MNIST_trainset(self): 
        train_set = get_training_dataset(self.BATCH_SIZE)
        one_batch = next(iter(train_set))
        idx = np.random.randint(self.BATCH_SIZE)
        img = one_batch[0][idx]
        label = one_batch[1][idx]

        self.assertIs(type(img), torch.Tensor)
        self.assertIs(type(label), torch.Tensor)
        self.assertIs(len(one_batch[0]), self.BATCH_SIZE)
        self.assertIs(len(one_batch[1]), self.BATCH_SIZE)
        self.assertAlmostEqual(img.mean().item(), 0, places=0)
        self.assertAlmostEqual(img.std().item(), 1, places=0)
    
    def test_MNIST_valset(self): 
        val_set = get_validation_dataset(self.BATCH_SIZE)
        one_batch = next(iter(val_set))
        idx = np.random.randint(self.BATCH_SIZE)
        img = one_batch[0][idx]
        label = one_batch[1][idx]

        self.assertIs(type(img), torch.Tensor)
        self.assertIs(type(label), torch.Tensor)
        self.assertIs(len(one_batch[0]), self.BATCH_SIZE)
        self.assertIs(len(one_batch[1]), self.BATCH_SIZE)
        self.assertAlmostEqual(img.mean().item(), 0, places=0)
        self.assertAlmostEqual(img.std().item(), 1, places=0)
    
    def test_MNIST_unbalanced_trainset(self): 
        val_set = get_validation_dataset(self.BATCH_SIZE, isUnbalanced_bool=True)
        one_batch = next(iter(val_set))
        idx = np.random.randint(self.BATCH_SIZE)
        label = one_batch[1][idx]

        self.assertEqual(len(torch.unique(one_batch[1])), 2)
        
        
if __name__ == "__main__":
    unittest.main()
