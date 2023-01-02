import torch
import torchvision

import PIL
from icecream import ic
# TODO
def classAcc(model, anch_embeddings, pos_embeddings, neg_embeddings, threshold:float, device:torch.device)->float:
    """
    TODO
    """
    ### Count positive
    BATCH_SIZE = len(pos_embeddings)
    count_P = 0
    model.eval()
    with torch.no_grad():    
        ### The operator 'aten::linalg_vector_norm' is not currently supported on the MPS backend
        d1 = torch.linalg.norm((anchor_embedding.to("cpu") - pos_embeddings.to("cpu")), dim=1).to(device)
        mask = d1 < threshold
        count_positive = torch.eq(target, mask).sum()
        
        acc = (count_positive/BATCH_SIZE)
    return acc.item()


if __name__ == "__main__":
    anchor = torch.rand(1, 128).repeat(32, 1)
    prediction_emb = anchor.clone() * 0.95
    # target = prediction.clone() * 0.95
    # ic(classAcc(prediction, target, 0.5, torch.device("cpu")))
    # anchor_embedding = anchor_embedding.repeat(BATCH_SIZE, 1)
    
    # d1 = torch.linalg.norm((prediction_emb - anchor), dim=1)
    # mask = d1 < 0.5
    # ic(mask)
    # count_positive = torch.eq(target, mask).sum()
    # acc = (count_positive/BATCH_SIZE)
    # ic(acc)