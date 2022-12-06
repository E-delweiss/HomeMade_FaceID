import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from loss import face_dataset

class TestLoss(unittest.TestCase):
    def __init__(self, TestLoss) -> None:
        super().__init__(TestLoss)
        self.SIZE = 448
        
    def test_BatchAllTripletLoss(self): 
        pass
        
if __name__ == "__main__":
    unittest.main()
