from configparser import ConfigParser

import torch
import torchvision
from torchinfo import summary

class ResNet18_custom(torch.nn.Module):
    def __init__(self, pretrained):
        super(ResNet18_custom, self).__init__()
        ### Load ResNet model
        self.res18 = torchvision.models.resnet18(pretrained = pretrained)

        ## Freeze backbone
        for param in self.res18.parameters():
            param.requires_grad = False

        ## Head part
        self.res18.fc = torch.nn.Linear(512, 1)

    def forward(self, x):
        output = self.res18(x)
        return torch.sigmoid(output)



def resNet18_custom(load_weights=False, **kwargs) -> ResNet18_custom:  
    config = ConfigParser()
    config.read("config.ini")

    weights = config.get("WEIGHTS", "weights")

    model = ResNet18_custom(**kwargs)
    if load_weights:
        model.load_state_dict(torch.load(weights))
    
    return model