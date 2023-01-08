import unittest
import os, sys
from pathlib import Path
from pprint import pprint

import numpy as np

import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from metrics import metrics

class TestMetrics(unittest.TestCase):
    def __init__(self, TestMetrics) -> None:
        super().__init__(TestMetrics)
        self.BATCH_SIZE = 32
        self.target = torch.LongTensor(self.BATCH_SIZE).random_(2)
        
    def test_metrics(self):
        prediction = self.target.clone()
        idx_FP, idx_FN = 0, 0
        while prediction[idx_FP] == 1 :
            idx_FP = torch.IntTensor(1).random_(self.BATCH_SIZE)
        while prediction[idx_FN] == 0 :
            idx_FN = torch.IntTensor(1).random_(self.BATCH_SIZE)
        
        prediction[idx_FP] = 0
        prediction[idx_FN] = 1

        metric_dict = metrics(model, pred_embeddings, target, threshold, device)
        
        print(prediction)
        
if __name__ == "__main__":
    unittest.main()
