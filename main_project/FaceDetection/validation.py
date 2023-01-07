import torch
from icecream import ic

def validation_loop(model, validation_dataset, device, ONE_BATCH=False):
    """
    Execute validation loop
    TODO
    Args:
        model (nn.Module)
            Yolo model.
        validation_dataset (Dataset) 
            Validation dataloader
        device (torch.device, optional)
            Running device. Defaults to cpu
        ONE_BATCH (bool, optional)
            For debugging or testing, permits to load only one batch. Default to False.

    Returns: 
    """
    print("|")
    print("| Validation...")
    model.eval()
    for (img, target) in validation_dataset:
        img, target = img.to(device), target.to(device)
        
        with torch.no_grad():
            ### prediction
            prediction = model(img)
            
            if ONE_BATCH is True:
                break

    return img, target, prediction



