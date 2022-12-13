import torch
import torchvision

import PIL
from icecream import ic

def classAcc(model, pred_embeddings:torch.Tensor, target:torch.Tensor, threshold:float, device:torch.device)->float:
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
        anchor_embedding = model(anchor_t)
        anchor_embedding = anchor_embedding.repeat(BATCH_SIZE, 1)
    
        d1 = torch.linalg.norm((anchor_embedding - pred_embeddings), dim=1)
        mask = d1 < threshold
        count_positive = torch.eq(target, d1).sum()
        
        acc = (count_positive/BATCH_SIZE)
    return acc.item()


if __name__ == "__main__":
    prediction = torch.rand(64)
    target = torch.randint(0,2,(64,))
    ic(classAcc(prediction, target))
