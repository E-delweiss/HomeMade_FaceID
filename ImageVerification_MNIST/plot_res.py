import glob
import random as rd
import os
from sklearn.manifold import TSNE

import PIL
import numpy as np
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import torch
import torchvision

from model import tnn
from MNIST_dataset import get_training_dataset, get_validation_dataset

# Define our own plot function
def scatter(x, labels, root, isUnbalanced_bool=False, subtitle=None, save_fig=False):
    """
    TODO
    """
    ### Calculate the number of classes
    num_classes = 10
    
    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color

    ### Create a seaborn scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[labels.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ## ---------------------------- ##
    
    ## Add label on top of each cluster ##
    if isUnbalanced_bool==False:
        idx2name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else: 
        idx2name = ['0','7']
        
    txts = []
    for i in idx2name:
        # Position of each label.
        xtext, ytext = np.median(x[labels == int(i), :], axis=0)
        txt = ax.text(xtext, ytext, i, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)        
                
    if subtitle:
        plt.suptitle(subtitle)
    
    if save_fig:
        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(os.path.join(root, str(subtitle)))


device = torch.device('cpu')
tsne = TSNE(random_state=0, learning_rate='auto', init='pca')
net = tnn(input_shape=(1,28,28), load_weights=True)
net = net.to(device)
dataset_test = get_validation_dataset(512, isUnbalanced_bool=False)

set_img, set_label = next(iter(dataset_test))
set_img, set_label = set_img.to(device), set_label.to(device)

net.eval()
with torch.no_grad():
    test_outputs = net(set_img)
    
test_tsne_embeds = tsne.fit_transform(test_outputs.cpu().detach().numpy())

scatter(
    test_tsne_embeds,
    set_label.cpu().numpy(),
    root='results',
    isUnbalanced_bool=False,
    subtitle=f'BatchAll MNIST distribution (test set)',
    save_fig=True
)

### For unbalanced dataset
# scatter(
#     test_tsne_embeds,
#     set_label.cpu().numpy(),
#     root='results',
#     isUnbalanced_bool=True,
#     subtitle=f'BatchAll MNIST distribution unbalanced (test set)',
#     save_fig=True
# )

