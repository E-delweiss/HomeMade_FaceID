import logging
import glob
from datetime import datetime
import pickle
from configparser import ConfigParser

from tqdm import tqdm
import PIL
import torch
import torchvision

def create_logging(prefix:str):
    """
    Create logging file.

    Args:
        prefix (str)
    """
    assert type(prefix) is str, TypeError

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = (
    '%(asctime)s ::%(levelname)s:: %(message)s')

    tm = datetime.now()
    tm = tm.strftime("%d%m%Y_%Hh%M")
    logging_name = 'logs/logging_'+prefix+'_'+tm+'.log'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format, datefmt='%d/%m/%Y %H:%M:%S',
        filemode="w",
        filename=(logging_name),
    )
    logging.info(f"Model is {prefix}")
    


def set_device(device, verbose=0)->torch.device:
    """
    Set the device to 'cpu', 'cuda' or 'mps'.

    Args:
        None.
    Return:
        device : torch.device
    """
    if device == 'cpu':
        device = torch.device('cpu')

    if device == 'cuda' and torch.cuda.is_available():
        torch.device('cuda')
    elif device == 'cuda' and torch.cuda.is_available() is False:
        logging.warning(f"Device {device} not available.")

    if device == 'mps' and torch.has_mps:
        # logging.warning(f"Device {device} currently not working well with PyTorch.")
        # device = torch.device('cpu')
        device = torch.device('mps')

    logging.info("Execute on {}".format(device))
    if verbose:
        print("\n------------------------------------")
        print(f"Execute script on - {device} -")
        print("------------------------------------\n")

    return device


def defineRanger(pt_file:str, num_epoch:int)->range:
    """
    Create a ranger for the training loop. 
    Handle a start != 0 for checkpoint.

    Args:
        pt_file (str)
            Pytorch checkpoint file.
        num_epoch (int)
            Number of epochs (modify the epoch to start with)
    """
    start_epoch = int(pt_file[:pt_file.find("epochs")][-3:])
    end_epoch = start_epoch + num_epoch
    ranger = range(start_epoch, end_epoch+1)
    return ranger


def save_model(model, prefix:str, current_epoch:int, save:bool):
    """
    Save Pytorch weights of the model. Set the name based on timeclock.

    Args:
        model (torch model)
            Training model.
        prefix (str)
            Used to create the pt file name.
        current_epoch (int)
            Used to create the pt file name.
        save (bool)
            If False, set a warning in log file.
    """
    if save:
        path = f"{prefix}_{current_epoch+1}epochs"
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        path = path+'_'+tm+'.pt'
        torch.save(model.state_dict(), path)
        print("\n")
        print("*"*5, "Model saved to {}.".format(path))
        logging.info("\n")
        logging.info("Model saved to {}.".format(path))
    else:
        logging.warning("No saving has been requested for model.")
    return


def save_losses(train_loss:dict, val_loss:dict, model_name:str, save:bool):
    """
    Save training en validation losses to pickle files.

    Args:
        train_loss (dict)
        val_loss (dict)
        model_name (str)
        save (bool)
    """
    if save:
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        train_path = f"train_results_{model_name}_{tm}.pkl"
        val_path = f"val_results_{model_name}_{tm}.pkl"
        
        with open(train_path, 'wb') as pkl:
            pickle.dump(train_loss, pkl)

        with open(val_path, 'wb') as pkl:
            pickle.dump(val_loss, pkl)
        
        logging.info("Training results saved to {}.".format(train_path))
        logging.info("Validation results saved to {}.".format(val_path))
    else:
        logging.warning("No saving has been requested for losses.")
    return


def tqdm_fct(training_dataset):
    """
    Set a tqdm progress bar.
    """
    return tqdm(enumerate(training_dataset),
                total=len(training_dataset),
                initial=1,
                desc="Training : image",
                ncols=100)



def mean_std_normalization()->tuple:
    """
    Get the mean and std of the dataset RGB channels.
    Note:
        mean/std of the whole dataset :
            mean=(0.4236, 0.3698, 0.3317), std=(0.2988, 0.2733, 0.2654)

    Returns:
        mean : torch.Tensor
        std : torch.Tensor
    """
    imgset_self = glob.glob('../dataset/dataset_moi_mbp_cropped/*')
    imgset_lfw = glob.glob('../dataset/lfw/lfw_funneled/*/*')
    data_jpg = imgset_self + imgset_lfw
    data_PIL = [PIL.Image.open(img_path).convert('RGB') for img_path in data_jpg]
    data_tensor = [torchvision.transforms.ToTensor()(img_PIL) for img_PIL in data_PIL]

    channels_sum, channels_squared_sum = 0, 0
    for img in data_tensor:
        channels_sum += torch.mean(img, dim=[1,2])
        channels_squared_sum += torch.mean(img**2, dim=[1,2])
    
    mean = channels_sum/len(data_tensor)
    std = torch.sqrt((channels_squared_sum/len(data_tensor) - mean**2))
    
    return mean, std

def unormalization(img_tensor:torch.Tensor):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.4236/0.2988, -0.4055/0.2733, -0.3317/0.2654],
        std=[1/0.2988, 1/0.2733, 1/0.2654]
        )
    img_idx = inv_normalize(img_tensor) * 255.0
    img_idx = img_tensor.to(torch.uint8)
    return img_idx


if __name__ == "__main__":
    mean, std = mean_std_normalization()
    print(mean)
    print(std)