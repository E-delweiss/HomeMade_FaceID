import torch
import torchvision

import PIL
from icecream import ic

def metrics(anchor_embedding:torch.Tensor, pred_embeddings:torch.Tensor, target:torch.Tensor, threshold:float, device:torch.device)->dict:
    """
    Compute metrics for the current training step.

    Args:
        anchor_embedding (torch.Tensor of size (BATCH_SIZE, 128))
            Anchor embedding repeated BATCH_SIZE times and created by the model. Since the model is training, 
            it changes each time the validation loop is called.
        pred_embeddings (torch.Tensor of size (BATCH_SIZE, 128))
            Prediction embeddings of the current batch.
        target (torch.Tensor of size (BATCH_SIZE, 1))
            Target of the current batch (0 if it is not the administrator's face, 1 if it is).
        threshold (float)
            Threshold for the distance between anchor and prediction embeddings. If the distance is lower, 
            the two images show the same person, else the persons are different.
        device (torch.device)

    Returns:
        metric_dict: {TN, TP, FP, FN, recall, precision, F1_score}
            Contain the computed metrics for the current training step.
    """
    ### The operator 'aten::linalg_vector_norm' is not currently supported on the MPS backend
    anchor_embedding, pred_embeddings, target = anchor_embedding.to("cpu"), pred_embeddings.to("cpu"), target.to("cpu")
    d1 = torch.linalg.norm((anchor_embedding - pred_embeddings), dim=1)

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