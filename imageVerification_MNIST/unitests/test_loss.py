import unittest
import os, sys
from pathlib import Path

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from loss import BatchAllTripletLoss

class TestLoss(unittest.TestCase):
    def __init__(self, TestLoss) -> None:
        super().__init__(TestLoss)
        self.SIZE = 28
        self.BATCH_SIZE = 32
        self.MARGIN = 10

    def test_loss(self):
        true_labels = torch.rand(self.BATCH_SIZE)
        pred_embedding = torch.rand(self.BATCH_SIZE, 128)
        pred_embedding2 = torch.rand(self.BATCH_SIZE).repeat(1,128)

        criterion = BatchAllTripletLoss(margin=self.MARGIN, device=torch.device('cpu'))
        loss, fraction_positive_triplets = criterion(pred_embedding, true_labels)
        loss2, fraction_positive_triplets2 = criterion(pred_embedding2, true_labels)

        self.assertIs(type(loss), torch.Tensor)
        self.assertIs(type(fraction_positive_triplets), torch.Tensor)
        self.assertEqual(loss2, 0)


if __name__ == "__main__":
    unittest.main()