import os
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


import torch
import torchvision
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

from model import siameseNet
from siamese_dataset_2faces import get_dataset
# from siamese_dataset_kfaces import get_dataset



# Define our own plot function
def scatter(x, labels, root='results', subtitle=None):
    
    num_classes = len(set(labels)) # Calculate the number of classes
    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color

    ## Create a seaborn scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    
    ## Add label on top of each cluster
    idx2name = ['0', '1']
        
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, idx2name[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)        
                
    if subtitle != None:
        plt.suptitle(subtitle)

    plt.savefig(os.path.join(root, str(subtitle)))


def plot_TSNEE(bacth_size, device, subtitle):
    net = siameseNet()
    net = net.to(device)

    dataloader = get_dataset(bacth_size, isNormalize_bool=True)
    set_img, set_label = next(iter(dataloader))
    set_img, set_label = set_img.to(device), set_label.to(device)

    tsne = TSNE(random_state=0, learning_rate='auto', init='pca')

    net.eval()
    with torch.no_grad():
        embeddings = net(set_img)
    test_tsne_embeds = tsne.fit_transform(embeddings.detach().numpy())

    scatter(test_tsne_embeds, set_label.numpy(), subtitle=subtitle)

def plot_confusionMatrix(bacth_size, threshold, device, root="results", title=None):
    mean, std = (0.3533, 0.3867, 0.5007), (0.2228, 0.2410, 0.2774)
    normalizer = torchvision.transforms.Normalize(mean, std)

    net = siameseNet()
    net = net.to(device)

    dataloader = get_dataset(bacth_size, isNormalize_bool=True)
    imgs_val, target = next(iter(dataloader))
    imgs_val, target = imgs_val.to(device), target.to(device)    

    ### Define the anchor image (i.e. the administrator's face)
    anchor_path = "../../dataset/frame_base_cropped.jpeg"
    anchor_PIL = PIL.Image.open(anchor_path).convert('RGB').resize((160,160))
    anchor_t = torchvision.transforms.ToTensor()(anchor_PIL).to(device)

    ### Preditions
    net.eval()
    with torch.no_grad():
        pred_embeddings = net(imgs_val)
        anchor_embedding = net(normalizer(anchor_t).unsqueeze(0)).repeat(len(target), 1)  

    ### Distance between pred_embeddings and anchor embedding
    d1 = torch.linalg.norm((anchor_embedding - pred_embeddings), dim=1)
    predictions = [1 if y_pred <= threshold else 0 for y_pred in d1]

    ax= plt.subplot()
    cm = confusion_matrix(target, predictions)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['notmyface', 'myface']); ax.yaxis.set_ticklabels(['notmyface', 'myface'])

    plt.savefig(os.path.join(root, str(title)))

if __name__ == "__main__":
    # plot_TSNEE(256, torch.device("cpu"), subtitle="InceptionResNet_vggface2_1vs1_faces")
    plot_confusionMatrix(256, threshold=1, device=torch.device("cpu"), root="results", title="Confusion_Matrix")








# ## Confusion matrix
# fig_confusionMatrix = plt.figure(figsize = (12,7))

# # constant for classes
# classes = [str(x) for x in range(C)]

# # Build confusion matrix
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*C, index = [i for i in classes],
#                      columns = [i for i in classes])

# sn.heatmap(df_cm, annot=True)
# plt.savefig('confusion_matrix.png')
