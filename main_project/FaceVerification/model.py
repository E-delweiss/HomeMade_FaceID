from configparser import ConfigParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from facenet_pytorch import InceptionResnetV1
from torchinfo import summary
from icecream import ic

class SiameseNet(nn.Module):
    def __init__(self, pretrained='vggface2'):
        """Custom the FaceNet InceptionResnetV1 model.
        """
        super(SiameseNet, self).__init__()
        # self.FaceNetModif_seq1 = nn.Sequential(*list(InceptionResnetV1(pretrained=pretrained).children())[:-3])
        self.FaceNetModif_seq1 = InceptionResnetV1(pretrained=pretrained)
        # self.FaceNetModif_seq2 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(1792, 128, bias=True),
        #     nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)
        # )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Forward method of the model.

        -------------------
        Parameters:
            x: torch.Tensor of shape (batch_size, channels, height, width)
                Batch of the tensor images
        -------------------
        Returns:
            x: torch.Tensor of shape (batch_size, 128)
                Batch of embeddings as the predictions of the model
        """
        x = self.FaceNetModif_seq1(x)
        # x = self.FaceNetModif_seq2(x)
        return x

def siameseNet(load_weights=False, **kwargs) -> SiameseNet: 
    config = ConfigParser()
    config.read("config.ini")

    weights = config.get("WEIGHTS", "weights")

    model = SiameseNet(**kwargs)
    if load_weights:
        model.load_state_dict(torch.load(weights))
        print("Load {}".format(weights))
    return model
    

if __name__ == "__main__":
    model = siameseNet()
    summary(model, (16, 1, 160, 160))
    