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

from model import tnn

class TestModel(unittest.TestCase):
    def __init__(self, TestModel) -> None:
        super().__init__(TestModel)
        self.SIZE = 28
        self.BATCH_SIZE = 32
        self.input_shape = (self.BATCH_SIZE, 1, self.SIZE, self.SIZE)
        self.print_summary = True

    def test_TNN(self):
        model = tnn(input_shape=self.input_shape[1:], load_weights=False)
        img_test = torch.rand(self.input_shape)
        output = model(img_test)
        
        self.assertEqual(output.shape, torch.Size([self.BATCH_SIZE, 128]))
        
        if self.print_summary:
            summary(model, input_size = self.input_shape)

if __name__ == "__main__":
    unittest.main()

