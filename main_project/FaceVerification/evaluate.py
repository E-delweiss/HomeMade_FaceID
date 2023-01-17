import datetime
import logging
import os
import sys
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision
import PIL

import utils
# from siamese_dataset import get_training_dataset, get_validation_dataset
from siamese_dataset import get_dataset
# from siamese_dataset_2faces import get_dataset
from model import siameseNet
from metrics import metrics
from loss import BatchAllTripletLoss


################################################################################
current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)
################################################################################

config = ConfigParser()
config.read('config.ini')

DEVICE = config.get('LOOP', 'device')
BATCH_SIZE = config.getint('LOOP', 'batch_size')
DO_METRICS = config.getboolean('LOOP', 'do_metrics')
ONE_BATCH = config.getboolean('LOOP', 'do_one_batch')

PREFIX = config.get('MODEL', 'model_name')
WEIGHTS = config.getboolean('MODEL', 'load_weights')
PRETRAINED = config.get('MODEL', 'pretrained')

TRESHOLD = config.getfloat('LOSS', 'threshold')

isNormalize = config.getboolean('DATASET', 'isNormalize')

FREQ = config.getint('PRINTING', 'freq')

################################################################################
device = utils.set_device(DEVICE, verbose=0)

model = siameseNet(load_weights=WEIGHTS, pretrained=PRETRAINED)
model = model.to(device)

dataloader = get_dataset(BATCH_SIZE, isNormalize_bool=isNormalize)
################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

time_formatted = datetime.datetime.now(tz=timezone)
time_formatted = '{:%Y-%m-%d %H:%M:%S}'.format(time_formatted)
start_time = datetime.datetime.now()

utils.create_logging(prefix=PREFIX)
logging.info(f"Pretrained is {PRETRAINED}")
if WEIGHTS: logging.info(f"WEIGHTS LOADED")
logging.info(f"Batch size = {BATCH_SIZE}")
logging.info(f"Threshold : {TRESHOLD}")
logging.info("")
logging.info("Start")
logging.info(f"[START] : {time_formatted}")

################################################################################
### Load anchor
anchor_path = "../../dataset/frame_base_cropped.jpeg"
anchor_PIL = PIL.Image.open(anchor_path).convert('RGB').resize((160,160))
anchor_t = torchvision.transforms.ToTensor()(anchor_PIL).to(device)
mean, std = (0.3533, 0.3867, 0.5007), (0.2228, 0.2410, 0.2774)
normalizer = torchvision.transforms.Normalize(mean, std)

metric_list = ["TP", "TN", "FP", "FN", "precision", "recall", "F1_score"]
metric_dict_val = dict.fromkeys(metric_list, 0)

model.eval()
for batch, (img, target) in utils.tqdm_fct(dataloader):
    with torch.no_grad():
        img, target = img.to(device), target.to(device)
        
        ### prediction
        pred_embeddings = model(img)
        anchor_embedding = model(normalizer(anchor_t).unsqueeze(0)).repeat(len(target), 1)  
        
        if ONE_BATCH is True:
            break

    if DO_METRICS:
        metric_dict = metrics(anchor_embedding, pred_embeddings, target, TRESHOLD, device)
        for key in metric_dict_val.keys():
            metric_dict_val[key] += metric_dict[key]

### Save accuracy
metric_dict_val.update((key, value/len(dataloader)) for key, value in metric_dict_val.items())
for metric in ["TP", "TN", "FP", "FN"]:
    logging.info(f"***** Mean {metric} : {metric_dict_val[metric]:.2f}")
for metric in ["precision", "recall"]:
    logging.info(f"***** Mean {metric} : {metric_dict_val[metric]:.2f}") 
logging.info(f"***** Mean F1_score : {metric_dict_val['F1_score']:.2f}")

################################################################################
end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
################################################################################