import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from dataset import face_dataset

class TestFaceDataset(unittest.TestCase):
    def __init__(self, TestFaceDataset) -> None:
        super().__init__(TestFaceDataset)
        self.SIZE = 448
        
    def test_face_dataset(self): 
        pass
        
if __name__ == "__main__":
    unittest.main()
