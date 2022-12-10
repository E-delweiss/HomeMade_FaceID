import datetime
import logging
import os, sys
from configparser import ConfigParser

import torch

import utils
from model import resNet18_custom
from facedetection_dataset import get_training_dataset, get_validation_dataset
from metrics import classAcc
from validation import validation_loop

################################################################################
current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)
################################################################################

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

FREQ = config.getint('PRINTING', 'freq')

################################################################################
device = utils.set_device(DEVICE, verbose=0)

model = resNet18_custom(pretrained=PRETRAINED, load_weights=LOAD_CHECKPOINT)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()

training_dataloader = get_training_dataset(BATCH_SIZE)
validation_dataloader = get_validation_dataset(VAL_BATCH_SIZE)

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
batch_train_loss = []
batch_val_acc = []

for epoch in ranger:
    epochs_loss = 0
    
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
        prediction = model(img)

        ### compute binary loss
        loss = criterion(prediction, target.unsqueeze(1))
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ######### print part #######################
        current_loss = loss.item()
        epochs_loss += current_loss
        print("\nCurrent loss : ", current_loss)
        if (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_train_loss.append(current_loss)

            ############### Compute validation metrics each FREQ batch ###########################################
            if DO_VALIDATION:
                _, target_val, prediction_val = validation_loop(model, validation_dataloader, device, ONE_BATCH=True)
                ### Validation accuracy
                acc = classAcc(prediction_val.squeeze(1), target_val)
                batch_val_acc.append(acc)

                print(f"| Validation class acc : {acc*100:.2f}%")
                print("\n\n")
            else : 
                acc = 9999
            ################################################################################

            if batch == 0 or (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
                print(f"\nMean training loss for this epoch : {epochs_loss / len(training_dataloader):.5f}")
                print("\n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"***** Training loss : {current_loss:.5f}")
                logging.info(f"***** Validation loss : {acc*100:.2f}%")

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

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
################################################################################