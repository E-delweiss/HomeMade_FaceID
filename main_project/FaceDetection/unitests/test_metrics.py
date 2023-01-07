import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision
from torchinfo import summary

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from metrics import classAcc

class TestMetrics(unittest.TestCase):
    def __init__(self, TestModel) -> None:
        super().__init__(TestModel)
        self.BATCH_SIZE = 32

    def test_classAcc(self):
        target = torch.randint(0, 2, (self.BATCH_SIZE, 1))
        prediction1 = torch.randint(0, 2, (self.BATCH_SIZE, 1))
        prediction2 = target.clone()
    
        self.assertIs(type(classAcc(prediction1, target)), float)
        self.assertLessEqual(classAcc(prediction1, target), 1.)        
        self.assertEqual(classAcc(prediction2, target), 1.)     

        
if __name__ == "__main__":
    unittest.main()

