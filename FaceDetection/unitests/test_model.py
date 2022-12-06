import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from model import SiameseNet

class TestModel(unittest.TestCase):
    def __init__(self, TestModel) -> None:
        super().__init__(TestModel)
        self.SIZE = 448
        
    def test_SiameseNet(self): 
        pass
        
if __name__ == "__main__":
    unittest.main()
