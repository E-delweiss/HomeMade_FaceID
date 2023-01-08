import torch
from icecream import ic
from metrics import metrics

def validation_loop(model, validation_dataset, device, do_metrics=False, ONE_BATCH=False):
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
    
    metric_list = ["TP", "TN", "FP", "FN", "precision", "recall", "F1_score"]
    metric_dict_val = dict.fromkeys(metric_list, 0)
    
    for (img_val, target_val) in validation_dataset:
        img_val, target_val = img_val.to(device), target_val.to(device)
        
        with torch.no_grad():
            ### prediction
            pred_embeddings_val = model(img_val)
            
            if ONE_BATCH is True:
                break

        if do_metrics:
            metric_dict = metrics(model, pred_embeddings_val, target_val, 0.5, device)
            for key in metric_dict_val.keys():
                metric_dict_val[key] += metric_dict[key]

    metric_dict_val.update((key, value/len(validation_dataset)) for key, value in metric_dict_val.items())

    return img_val, target_val, pred_embeddings_val, metric_dict_val



