import torch
from icecream import ic

def classAcc(prediction:torch.Tensor, target:torch.Tensor)->float:
    """
    TODO
    """
    BATCH_SIZE = len(target)
    acc_sum = torch.sum(torch.eq(torch.round(prediction),target)).item()
    acc = acc_sum / BATCH_SIZE
    return acc