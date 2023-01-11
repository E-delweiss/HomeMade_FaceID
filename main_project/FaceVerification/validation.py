import torch
from icecream import ic
from metrics import metrics

import PIL
import torch
import torchvision 

def validation_loop(model, validation_dataset, device, do_metrics=False, threshold=None, ONE_BATCH=False):
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

    #######################################################################################
    ### Load anchor
    mean, std = (0.3533, 0.3867, 0.5007), (0.2228, 0.2410, 0.2774)
    anchor_path = "../../dataset/frame_base_cropped.jpeg"
    anchor_PIL = PIL.Image.open(anchor_path).convert('RGB').resize((160,160))
    anchor_t = torchvision.transforms.ToTensor()(anchor_PIL)
    anchor_norm_t = torchvision.transforms.Normalize(mean, std)(anchor_t).to(device)

    #######################################################################################
    metric_list = ["TP", "TN", "FP", "FN", "precision", "recall", "F1_score"]
    metric_dict_val = dict.fromkeys(metric_list, 0)
    
    model.eval()
    for (img_val, target_val) in validation_dataset:
        img_val, target_val = img_val.to(device), target_val.to(device)
        with torch.no_grad():
            ### prediction
            pred_embeddings_val = model(img_val)
            anchor_embedding = model(anchor_norm_t.unsqueeze(0)).repeat(len(target_val), 1)  
            
            if ONE_BATCH is True:
                break

        if do_metrics:
            metric_dict = metrics(anchor_embedding, pred_embeddings_val, target_val, threshold, device)
            # ic(metric_dict)
            for key in metric_dict_val.keys():
                metric_dict_val[key] += metric_dict[key]

    metric_dict_val.update((key, value/len(validation_dataset)) for key, value in metric_dict_val.items())

    return img_val, target_val, pred_embeddings_val, metric_dict_val



