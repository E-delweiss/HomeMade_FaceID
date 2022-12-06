import datetime
import logging
import os
import sys
from configparser import ConfigParser

import torch


################################################################################
current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)
################################################################################

config = ConfigParser()
config.read('config.ini')

DEVICE = config.get('TRAINING', 'device')
learning_rate = config.getfloat('TRAINING', 'learning_rate')
BATCH_SIZE = config.getint('TRAINING', 'batch_size')
WEIGHT_DECAY = config.getfloat('TRAINING', 'weight_decay')
DO_VALIDATION = config.getboolean('TRAINING', 'do_validation')
EPOCHS = config.getint('TRAINING', 'nb_epochs')
LR_SCHEDULER = config.getboolean('TRAINING', 'lr_scheduler')

SAVE_MODEL = config.getboolean('SAVING', 'save_model')
SAVE_LOSS = config.getboolean('SAVING', 'save_loss')

PREFIX = config.get('MODEL', 'model_name')
PRETRAINED = config.getboolean('MODEL', 'pretrained_resnet')
LOAD_CHECKPOINT = config.getboolean('MODEL', 'yoloResnet_checkpoint')

isNormalize_trainset = config.getboolean('DATASET', 'isNormalize_trainset')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isNormalize_valset = config.getboolean('DATASET', 'isNormalize_valset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

FREQ = config.getint('PRINTING', 'freq')

################################################################################
device = utils.set_device(DEVICE, verbose=0)

model = yoloResnet(resnet_pretrained=PRETRAINED, load_yoloweights=LOAD_CHECKPOINT, S=S, B=B, C=C)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
criterion = YoloLoss(lambd_coord=LAMBD_COORD, lambd_noobj=LAMBD_NOOBJ, S=S, device=device)

training_dataloader = get_training_dataset(BATCH_SIZE, split="train", isNormalize=isNormalize_trainset, isAugment=isAugment_trainset)
validation_dataloader = get_validation_dataset(split="test", isNormalize=isNormalize_valset, isAugment=isAugment_valset)

if LOAD_CHECKPOINT:
    pt_file = config.get('WEIGHTS', 'resnetYolo_weights')
    ranger = utils.defineRanger(pt_file, EPOCHS)
else:
    ranger = range(EPOCHS)
################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

time_formatted = datetime.datetime.now(tz=timezone)
time_formatted = '{:%Y-%m-%d %H:%M:%S}'.format(time_formatted)
start_time = datetime.datetime.now()

print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")

utils.create_logging(prefix=PREFIX)
logging.info(f"Pretrained is {PRETRAINED}")
if LOAD_CHECKPOINT: logging.info(f"RESTART FROM CHECKPOINT")
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")
logging.info(f"Using optimizer : {optimizer}")
logging.info("Lr Scheduler : None")
logging.info("")
logging.info("Start training")
logging.info(f"[START] : {time_formatted}")

################################################################################
batch_total_train_loss_list = []
batch_train_losses_list = []
batch_train_class_acc = []

batch_val_MSE_box_list = []
batch_val_confscore_list = []
batch_val_class_acc = []

all_pred_boxes = []
all_true_boxes = []
for epoch in ranger:
    TODO
