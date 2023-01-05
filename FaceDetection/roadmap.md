# Building a Face Detection model

## Overview
This model comes along my "FaceID project". It's the first part of this project which should classify if a photo contains a face or no. 

It's a binary classification and uses the weights of a ResNet50 model pretrained on [ImageNet](https://image-net.org/) and then, fine tuned with faces drawn from the [LFW database](http://vis-www.cs.umass.edu/lfw/) and backgrounds drawn from [House Rooms dataset](https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset). 

This model is known as a *face detection* model (opposed to a *face recognition* model which will see in detail in the next part). It classifies whether there is a face (a person) or not in a image. The output of the model is binary (0 or 1) where 0 means there is no face and 1 there is a face.

## Dataset
I chose to augment the dataset with random horizontal flips and random rotations to simulate subject's positionning imperfections.

Also, as suggests the [ResNet paper](https://arxiv.org/abs/1512.03385), and as recalls the [Pytorch websibe](https://pytorch.org/hub/pytorch_vision_resnet/): when using pretrained ResNet, image pixel values should be between `[0,1]` and one should normalize the images using the following values `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.

## Model
I used a pretrained ResNet50 model from torchvision models. All the layers are frozen (`param.requires_grad = False`) and the last layer is turned into a linear layer (`torch.nn.Linear`) with a unique output unit (binary classification). 

Finally, a sigmoid function to the forward output is applied to turn the output into probability.

## Compilation
* Loss: `torch.nn.BCELoss` which measures the Binary Cross Entropy between the target and the input probabilities
* Optimizer: `torch.optim.Adam` with `lr=0.001`

## Notes
This face detector lacking of generalization when comes to no background or over exposed camera view (respectively black and white images). A preprocessing could be done to ignore those cases or analysing the early activation filters when there are those exception and when there are not.
