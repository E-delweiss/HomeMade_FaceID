from configparser import ConfigParser

import torch
import torch.nn as nn
import torchvision

class ResNet50_custom(torch.nn.Module):
    def __init__(self):
        super(resNet50_custom, self).__init__()
        self.res50 = torchvision.models.resnet50(pretrained = True, progress = True)
        for param in self.res50.parameters():
            param.requires_grad = False
        self.res50.fc = torch.nn.Linear(2048, 1)

    def forward(self, input):
        output = self.res50(input)
        return torch.sigmoid(output)



def resNet50_custom(load_weights=False, pretrained=True, **kwargs) -> ResNet50_custom:  
    config = ConfigParser()
    config.read("config.ini")

    weights = config.get("WEIGHTS", "weights")

    model = ResNet50_custom(**kwargs)
    if load_weights:
        model.load_state_dict(torch.load(weights))
    
    return model
    

if __name__ == "__main__":
    model = resNet50_custom(???)
    summary(model, (16, 3, 448, 448))
    