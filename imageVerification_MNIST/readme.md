# Introduction
This GitHub repo is about face verification. To tackle this subject, I chose to start with the MNIST dataset. The goal here is to understand how to use the *Siamese Network* method. At the end, each digit of the MNIST dataset is expected to be encoding into a 128d embedding vector where each digit embedding vector is grouped in a 128d space with its fellows.

Two aspects of this project are depicted:
* First, train a model that takes each digit apart **from each other**
* Second, be able to deal with an unbalanced dataset where the model should be able to take apart **an only digit from all the other**.

Note: all the concepts won't be explain in detail here but rather in the [main project's wiki](https://github.com/E-delweiss/HomeMade_FaceID/wiki/Face-verification).

# Dataset
The MNIST dataset from `torchvision` is used. Each image is turned into `torch.Tensor` and normalize with `(mean, std) = (0.1307, 0.3081)`. No data augmentation has been used.

To treat the *unbalanced* case, the **7 digit** has been chosen so, when the dataloader calls the dataset class, an image of a 7 has the label 7 but the label of any other image will be 0.

# Model
This task is not really complicated for a network. Here I used a simple network called TNN and find in [this repo](https://github.com/KinWaiCheuk/pytorch-triplet-loss/blob/master/TNN/Model.py) from @KinWaiCheuk.

# Compilation
* Loss: custom loss `BatchAllTripletLoss` with the *online mining all* technique (see the [main project's roadmap](https://github.com/E-delweiss/HomeMade_FaceID/wiki/Face-verification))
* Optimizer: `torch.optim.Adam` with `lr=0.0001` and `weight_decay=0.0005`

# Results
I didn't try to extract any accuracy here but only a qualitative result that shows how this model can encode MNIST digits such as each class are grouped together in the embedding space (TSNE vizualisation).

<p align="center">
  <img src="https://github.com/E-delweiss/HomeMade_FaceID/blob/main/imageVerification_MNIST/results/BatchAll MNIST distribution (test set).png?raw=true" alt="balanced_set" width="300"/>
  <img src="https://github.com/E-delweiss/HomeMade_FaceID/blob/main/imageVerification_MNIST/results/BatchAll MNIST distribution unbalanced (test set).png?raw=true" alt="unbalanced_set" width="300"/>
</p>
