import torchvision
import torch
import PIL
from datetime import datetime
import numpy as np
from icecream import ic
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os

# Define our own plot function
def scatter(x, labels, root='plot', subtitle=None):
    
    num_classes = len(set(labels)) # Calculate the number of classes
    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color

    ## Create a seaborn scatter plot ##
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ## ---------------------------- ##
    
    ## Add label on top of each cluster ##
    idx2name = ['0', '1', '2']
        
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, idx2name[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)        
        
    ## ---------------------------- ##    
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(os.path.join(root, str(subtitle)))

def TSNEE_plot(model, dataloader):
    tm = datetime.now()
    tm = tm.strftime("%d%m_%Hh%M_%Ss")
    device = next(model.parameters()).device

    img, target = next(iter(dataloader))
    img, target = img.to(device), target.to(device)

    anchor_path = "../../dataset/frame_base_cropped.jpeg"
    anchor_PIL = PIL.Image.open(anchor_path).convert('RGB').resize((160,160))
    anchor_t = torchvision.transforms.ToTensor()(anchor_PIL).to(device)

    tsne = TSNE(random_state=0, learning_rate='auto', init='pca')

    model.eval()
    with torch.no_grad():
        pred_outputs = model(torch.cat([img, anchor_t.unsqueeze(0)], axis=0))
    tsne_embeds = tsne.fit_transform(pred_outputs.cpu().detach().numpy())

    targets = torch.cat([target, torch.tensor([2]).to(device)],axis=0).cpu().numpy()
    scatter(tsne_embeds, targets, subtitle=f'TSNEE_{tm}')