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


if __name__ == "__main__":
    prediction = torch.rand(64)
    target = torch.randint(0,2,(64,))
    ic(classAcc(prediction, target))