import datetime
import logging
import os
import sys
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter

import torch

import utils
from siamese_dataset import get_training_dataset, get_validation_dataset
from model import siameseNet
from metrics import classAcc
from loss import BatchAllTripletLoss
from validation import validation_loop


################################################################################
current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)
################################################################################

writer = SummaryWriter()
config = ConfigParser()
config.read('config.ini')

DEVICE = config.get('TRAINING', 'device')
learning_rate = config.getfloat('TRAINING', 'learning_rate')
BATCH_SIZE = config.getint('TRAINING', 'train_batch_size')
VAL_BATCH_SIZE = config.getint('TRAINING', 'val_batch_size')
WEIGHT_DECAY = config.getfloat('TRAINING', 'weight_decay')
DO_VALIDATION = config.getboolean('TRAINING', 'do_validation')
EPOCHS = config.getint('TRAINING', 'nb_epochs')
LR_SCHEDULER = config.getboolean('TRAINING', 'lr_scheduler')

SAVE_MODEL = config.getboolean('SAVING', 'save_model')
SAVE_LOSS = config.getboolean('SAVING', 'save_loss')

PREFIX = config.get('MODEL', 'model_name')

PRETRAINED = config.getboolean('MODEL', 'pretrained')
LOAD_CHECKPOINT = config.getboolean('MODEL', 'checkpoint')

MARGIN = config.getfloat('LOSS', 'margin')
TRESHOLD = config.getfloat('LOSS', 'threshold')

RATIO = config.getint('DATASET', 'ratio')
isNormalize_trainset = config.getboolean('DATASET', 'isNormalize_trainset')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isNormalize_valset = config.getboolean('DATASET', 'isNormalize_valset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

FREQ = config.getint('PRINTING', 'freq')

################################################################################
device = utils.set_device(DEVICE, verbose=0)

model = siameseNet(load_weights=LOAD_CHECKPOINT)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
criterion = BatchAllTripletLoss(margin=MARGIN, device=device)

training_dataloader = get_training_dataset(BATCH_SIZE, ratio=RATIO, isNormalize_bool=isNormalize_trainset, isAugment_bool=isAugment_trainset)
validation_dataloader = get_validation_dataset(VAL_BATCH_SIZE, ratio=1, isNormalize_bool=isNormalize_valset, isAugment_bool=isAugment_valset)

if LOAD_CHECKPOINT:
    pt_file = config.get('WEIGHTS', 'weights')
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
for epoch in ranger:
    epochs_loss = 0
    batch_train_loss = []
    batch_val_acc = []
    
    print("-"*20)
    print(" "*5 + f"EPOCH {epoch+1}/{EPOCHS+ranger[0]}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    ################################################################################
    
    for batch, (img, target) in utils.tqdm_fct(training_dataloader):    
        model.train()
        img, target = img.to(device), target.to(device)
        
        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction
        pred_embeddings = model(img)

        ### compute binary loss
        loss, fraction_positive_triplets = criterion(pred_embeddings, target)
        writer.add_scalar("Loss/train", loss, (epoch+1)*batch)
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        current_loss = loss.item()
        epochs_loss += current_loss
        # print("\nCurrent loss : ", current_loss)
        if (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_train_loss.append(current_loss)

            ############### Compute validation metrics each FREQ batch ###########################################
            if DO_VALIDATION:
                _, target_val, pred_embeddings_val = validation_loop(model, validation_dataloader, device, ONE_BATCH=False)
                
                ### Validation accuracy
                print("Compute acc...")      
                acc = classAcc(model, pred_embeddings_val, target_val, TRESHOLD, device)
                writer.add_scalar("Accuracy/val", acc, (epoch+1)*batch)
                batch_val_acc.append(acc)
                

            else : 
                acc = 9999
            ################################################################################

            if batch == 0 or (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"Fraction positive triplet in this batch : {fraction_positive_triplets*100:.2f}")
                logging.info(f"***** Training loss : {current_loss:.5f}")
                logging.info(f"***** Validation accuracy : {acc*100:.2f}%")
            
            if batch == len(training_dataloader.dataset)//BATCH_SIZE:
                logging.info(f"Mean training loss for this epoch : {epochs_loss / len(training_dataloader):.5f}")
                print(f"\nMean training loss for this epoch : {epochs_loss / len(training_dataloader):.5f}")
                logging.info("")
                print("\n\n")
################################################################################
### Saving results
pickle_val_results = {
"batch_val_acc":batch_val_acc
}

pickle_train_results = {
    "batch_train_loss" : batch_train_loss,
}

utils.save_model(model, PREFIX, epoch, SAVE_MODEL)
utils.save_losses(pickle_train_results, pickle_val_results, PREFIX, SAVE_LOSS)

writer.flush()
end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
################################################################################