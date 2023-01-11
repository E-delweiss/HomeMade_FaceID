import torch
import torchvision

import PIL
from icecream import ic

def metrics(anchor_embedding, pred_embeddings:torch.Tensor, target:torch.Tensor, threshold:float, device:torch.device)->float:
    """
    TODO
    """
    ### The operator 'aten::linalg_vector_norm' is not currently supported on the MPS backend
    anchor_embedding, pred_embeddings, target = anchor_embedding.to("cpu"), pred_embeddings.to("cpu"), target.to("cpu")
    d1 = torch.linalg.norm((anchor_embedding - pred_embeddings), dim=1)

    # ic(d1)
    TP = (d1 < threshold) & (target == 1)
    TN = (d1 >= threshold) & (target == 0)
    FP = (d1 < threshold) & (target == 0)
    FN = (d1 >= threshold) & (target == 1)
    
    count_TP = TP.sum()
    count_TN = TN.sum()
    count_FP = FP.sum()
    count_FN = FN.sum()    
    
    precision = count_TP / (count_TP + count_FP + 1e-6)
    recall = count_TP / (count_TP + count_FN + 1e-6)
    F1_score = 2*(precision*recall) / (precision + recall + 1e-6)

    metric_dict = {
        "TP" : count_TP,
        "TN" : count_TN, 
        "FP" : count_FP,
        "FN" : count_FN,
        "recall" : recall,
        "precision" : precision,
        "F1_score" : F1_score
    }
    return metric_dict


if __name__ == "__main__":
    anchor = torch.rand(1, 128).repeat(32, 1)
    prediction_emb = anchor.clone() * 0.95