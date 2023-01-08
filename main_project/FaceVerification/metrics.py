import torch
import torchvision

import PIL
from icecream import ic

def metrics(model, pred_embeddings:torch.Tensor, target:torch.Tensor, threshold:float, device:torch.device)->float:
    """
    TODO
    """
    ### Load anchor
    anchor_path = "../dataset/frame_base_cropped.jpeg"
    anchor_PIL = PIL.Image.open(anchor_path).convert('RGB').resize((160,160))
    anchor_t = torchvision.transforms.ToTensor()(anchor_PIL)

    ### Count positive
    BATCH_SIZE = len(target)
    count_P = 0
    model.eval()
    with torch.no_grad():
        anchor_t = anchor_t.unsqueeze(0).to(device)
        anchor_embedding = model(anchor_t).repeat(BATCH_SIZE, 1)
    
    ### The operator 'aten::linalg_vector_norm' is not currently supported on the MPS backend
    d1 = torch.linalg.norm((anchor_embedding.to("cpu") - pred_embeddings.to("cpu")), dim=1).to(device)

    mask_TP = (d1 < threshold) & (target == 1)
    mask_TN = (d1 >= threshold) & (target == 0)
    mask_FP = (d1 < threshold) & (target == 0)
    mask_FN = (d1 >= threshold) & (target == 1)
    
    count_TP = target[mask_TP].sum()
    count_TN = target[mask_TN].sum()
    count_FP = target[mask_FP].sum()
    count_FN = target[mask_FN].sum()    
    
    precision = count_TP / (count_TP + count_FP)
    recall = count_TP / (count_TP + count_FN)
    F1_score = 2*(precision*recall) / (precision + recall)

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