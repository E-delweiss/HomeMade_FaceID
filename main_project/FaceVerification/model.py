from configparser import ConfigParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from facenet_pytorch import InceptionResnetV1
from torchinfo import summary
from icecream import ic

class SiameseNet(nn.Module):
    def __init__(self, pretrained):
        """Custom the FaceNet InceptionResnetV1 model.
        """
        super(SiameseNet, self).__init__()
        count=0
        print(type(pretrained))
        assert pretrained in ['None', 'vggface2', 'casia-webface']
        pretrained = None if pretrained == 'None' else pretrained
        self.model = InceptionResnetV1(pretrained=pretrained)

        # if pretrained is not None:
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        #         # if count == 376:
        #         if count == 377:
        #             break
        #         count += 1

        # if pretrained == 'None':
        #     self.model = torch.nn.Sequential(*list(InceptionResnetV1(pretrained=None).children())[:-2])
        # else: 
        #     self.model = torch.nn.Sequential(*list(InceptionResnetV1(pretrained=pretrained).children())[:-3])

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.avgpool_1a.requires_grad_()
        self.model.dropout.requires_grad_()
        self.model.last_linear.requires_grad_()

        # self.model[-3].requires_grad_()
        # self.model[-2].requires_grad_()
        # self.model[-1].requires_grad_()

        # self.add_head = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=1792, out_features=512, bias=False),
        #     torch.nn.LayerNorm(512)
        #     )

    def forward(self, input:torch.Tensor)->torch.Tensor:
        """Forward method of the model.

        -------------------
        Parameters:
            input: torch.Tensor of shape (batch_size, channels, height, width)
                Batch of the tensor images
        -------------------
        Returns:
            x: torch.Tensor of shape (batch_size, 128)
                Batch of embeddings as the predictions of the model
        """
        x = self.model(input)
        # ic(x.shape)
        # x = self.add_head(torch.nn.Flatten()(x))
        return x


def siameseNet(load_weights=False, **kwargs) -> SiameseNet: 
    config = ConfigParser()
    config.read("config.ini")

    weights = config.get("WEIGHTS", "weights")

    model = SiameseNet(**kwargs)
    # model = TNN(**kwargs)
    if load_weights:
        model.load_state_dict(torch.load(weights))
        print("Load {}".format(weights))

    
    return model
    

if __name__ == "__main__":
    model = siameseNet(pretrained="vggface2")
    dummy = torch.rand(32, 3, 160, 160)
    output = model(dummy)
    print(output.shape)
    summary(model, (16, 3, 160, 160))
    