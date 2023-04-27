import torch
from icecream import ic
from metrics import metrics

import PIL
import torch
import torchvision 

def validation_loop(model, validation_dataset, device, do_metrics=False, threshold=None, ONE_BATCH=False):
    """
    _summary_

    Args:
        model (torch.nn.Module):
             Model to evaluate.
        validation_dataset (torch.utils.data.DataLoader)
        device (torch.device)
        do_metrics (bool, optional): 
            Compute the metrics during validation loop or not. Defaults to False.
        threshold (_type_, optional): 
            Threshold for the distance between anchor and prediction embeddings. If the distance is lower, 
            the two images show the same person, else the persons are different. 
            Used in the metrics.py module. Defaults to None.
        ONE_BATCH (bool, optional): 
            For debuging, returns a single batch for the validation dataset. Defaults to False.

    Returns:
        img_val (torch.Tensor of size (BATCH_SIZE, 3, 160, 160)): 
        target_val (torch.Tensor of size (BATCH_SIZE, 1)):
        pred_embeddings_val (torch.Tensor of size (BATCH_SIZE, 128): 
        metric_dict_val (dict{TN, TP, FP, FN, recall, precision, F1_score}
            Contain the computed metrics for the current training step.
            If do_metrics is True, metric_dict_val contains metrics with values
            equal to zero.

    """
    print("|")
    print("| Validation...")

    #######################################################################################
    ### Load anchor
    # mean, std = (0.3533, 0.3867, 0.5007), (0.2228, 0.2410, 0.2774)
    anchor_path = "../../dataset/frame_base_cropped.jpeg"
    anchor_PIL = PIL.Image.open(anchor_path).convert('RGB').resize((160,160))
    anchor_t = torchvision.transforms.ToTensor()(anchor_PIL).to(device)
    # anchor_norm_t = torchvision.transforms.Normalize(mean, std)(anchor_t).to(device)

    #######################################################################################
    metric_list = ["TP", "TN", "FP", "FN", "precision", "recall", "F1_score"]
    metric_dict_val = dict.fromkeys(metric_list, 0)
    
    model.eval()
    # for (img_val, target_val) in validation_dataset:
    for img_pos, img_neg, targ_pos, targ_neg in validation_dataset:
        # img, target = img.to(device), target.to(device)
        img_pos, img_neg, targ_pos, targ_neg = img_pos.to(device), img_neg.to(device), targ_pos.to(device), targ_neg.to(device)
        with torch.no_grad():
            ### prediction
            # pred_embeddings_val = model(img_val)
            pred_embeddings_pos = model(img_pos)
            pred_embeddings_neg = model(img_neg)
            anchor_embedding = model(anchor_t.unsqueeze(0))
            # ic(anchor_embedding.shape)

            pred_embeddings = torch.cat([pred_embeddings_pos, pred_embeddings_neg], axis=0)
            targets = torch.cat([targ_pos, targ_neg], axis=0)

            anchor_embedding_batch = anchor_embedding.repeat(pred_embeddings.shape[0], 1) 
            
            if ONE_BATCH is True:
                break

        if do_metrics:
            metric_dict = metrics(anchor_embedding_batch, pred_embeddings, targets, threshold, device)
            # ic(metric_dict)
            for key in metric_dict_val.keys():
                metric_dict_val[key] += metric_dict[key]

    metric_dict_val.update((key, value/len(validation_dataset)) for key, value in metric_dict_val.items())

    # return img_val, target_val, pred_embeddings_val, metric_dict_val
    return img_pos, img_neg, targets, anchor_embedding, metric_dict_val



