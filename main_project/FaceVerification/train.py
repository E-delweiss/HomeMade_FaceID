import datetime
import logging
import os
import sys
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision
import PIL
from torchinfo import summary

import utils
from siamese_dataset import get_training_dataset, get_validation_dataset
# from siamese_dataset_2faces import get_training_dataset, get_validation_dataset
from model import siameseNet
from metrics import metrics
from loss import BatchAllTripletLoss
from validation import validation_loop
from TSNEE_validation import TSNEE_plot


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
DO_TSNEE_PLOT = config.getboolean('TRAINING', 'do_tsnee')
DO_METRICS = config.getboolean('TRAINING', 'do_metrics')
EPOCHS = config.getint('TRAINING', 'nb_epochs')
LR_SCHEDULER = config.getboolean('TRAINING', 'lr_scheduler')

SAVE_MODEL = config.getboolean('SAVING', 'save_model')
SAVE_LOSS = config.getboolean('SAVING', 'save_loss')

PREFIX = config.get('MODEL', 'model_name')

PRETRAINED = config.get('MODEL', 'pretrained')
LOAD_CHECKPOINT = config.getboolean('MODEL', 'checkpoint')

MARGIN = config.getfloat('LOSS', 'margin')
TRESHOLD = config.getfloat('LOSS', 'threshold')

ratio = config.getint("DATASET", "ratio")
isNormalize_trainset = config.getboolean('DATASET', 'isNormalize_trainset')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isNormalize_valset = config.getboolean('DATASET', 'isNormalize_valset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

FREQ = config.getint('PRINTING', 'freq')

################################################################################
device = utils.set_device(DEVICE, verbose=0)

model = siameseNet(load_weights=LOAD_CHECKPOINT, pretrained=PRETRAINED)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
# criterion = BatchAllTripletLoss(margin=MARGIN, device=device)
criterion = torch.nn.TripletMarginLoss(margin=MARGIN)
# dummy = torch.rand(8, 3, 160, 160).to(device)
# output = model(dummy)
# summary(model.to("cpu"), (8, 3, 160, 160))

training_dataloader = get_training_dataset(BATCH_SIZE, ratio=ratio, isNormalize_bool=isNormalize_trainset, isAugment_bool=isAugment_trainset)
validation_dataloader = get_validation_dataset(VAL_BATCH_SIZE, ratio=ratio, isNormalize_bool=isNormalize_valset, isAugment_bool=isAugment_valset)

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

torch.autograd.set_detect_anomaly(True)

writer = SummaryWriter(log_dir=f'runs/{PREFIX}_RUN_{time_formatted}')

utils.create_logging(prefix=PREFIX)
logging.info(f"Pretrained is {PRETRAINED}")
if LOAD_CHECKPOINT: logging.info(f"RESTART FROM CHECKPOINT")
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")
logging.info(f"Using optimizer : {optimizer}")
logging.info("Lr Scheduler : None")
logging.info(f"Margin : {MARGIN}")
logging.info(f"Threshold : {TRESHOLD}")
logging.info("")
logging.info("Start training")
logging.info(f"[START] : {time_formatted}")

################################################################################
anchor_PIL = PIL.Image.open("/Users/thierryksstentini/Documents/Python_divers/GitHub/HomeMade_FaceID/dataset/frame_base_cropped.jpeg").convert('RGB').resize((160,160))
anchor_t = torchvision.transforms.ToTensor()(anchor_PIL).unsqueeze(0)

for epoch in ranger:
    epochs_loss = 0
    batch_train_loss = []
    batch_val_acc = []
    
    print("-"*20)
    print(" "*5 + f"EPOCH {epoch+1}/{EPOCHS+ranger[0]}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    ################################################################################
    
    # for batch, (img, target) in utils.tqdm_fct(training_dataloader):
    for batch, (img_pos, img_neg, targ_pos, targ_neg) in utils.tqdm_fct(training_dataloader):
        model.train()
        # img, target = img.to(device), target.to(device)
        img_pos, img_neg, targ_pos, targ_neg = img_pos.to(device), img_neg.to(device), targ_pos.to(device), targ_neg.to(device)

        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction
        with torch.no_grad():
            anchor_t_batch = anchor_t.repeat(img_pos.shape[0],1,1,1).to(device)
            pred_embeddings_anch = model(anchor_t_batch)
        pred_embeddings_neg = model(img_neg)
        pred_embeddings_pos = model(img_pos)


        ### compute binary loss
        # loss, fraction_positive_triplets = criterion(pred_embeddings, 
        # target)
        loss = criterion(pred_embeddings_anch, pred_embeddings_pos, pred_embeddings_neg)
        fraction_positive_triplets = 99999
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        current_loss = loss.item()
        epochs_loss += current_loss
        if (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_train_loss.append(current_loss)

            ############### Compute validation metrics each FREQ batch ###########################################
            if DO_VALIDATION:
                _, _, _, _, metric_dict_val = validation_loop(model, validation_dataloader, device, do_metrics=DO_METRICS, threshold=TRESHOLD, ONE_BATCH=False)
                
                ### Validation accuracy
                for metric in ["TP", "TN", "FP", "FN"]:
                    writer.add_scalars('variables', {f"{metric}/val" : metric_dict_val[metric]}, (epoch+1)*batch)               
                    logging.info(f"***** {metric} : {metric_dict_val[metric]:.2f}")
                for metric in ["precision", "recall"]:
                    writer.add_scalars('variables', {f"{metric}/val" : metric_dict_val[metric]}, (epoch+1)*batch)               
                    logging.info(f"***** {metric} : {metric_dict_val[metric]:.2f}") 
                writer.add_scalars('variables', {f"F1_score/val" : metric_dict_val["F1_score"]}, (epoch+1)*batch)               
                logging.info(f"***** F1_score : {metric_dict_val['F1_score']:.2f}")

            else : 
                acc = 9999

            if DO_TSNEE_PLOT:
                TSNEE_plot(model, validation_dataloader)
            ################################################################################

            if batch == 0 or (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"Fraction positive triplet in this batch : {fraction_positive_triplets*100:.2f}")
                logging.info(f"***** Training loss : {current_loss:.5f}")
            
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

utils.save_model(model, PREFIX, epoch, SAVE_MODEL, LOAD_CHECKPOINT)
utils.save_losses(pickle_train_results, pickle_val_results, PREFIX, SAVE_LOSS)

writer.flush()
writer.close()
end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
################################################################################