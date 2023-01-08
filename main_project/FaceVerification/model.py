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
        self.FaceNetModif_seq1 = nn.Sequential(*list(InceptionResnetV1(pretrained=pretrained).children())[:-3])
        self.FaceNetModif_seq2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792, 128, bias=True),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)
        )
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
        x = self.FaceNetModif_seq2(x)
        return x


class TNN(nn.Module):
    def __init__(self, input_shape=(), output_size=128, pretrained=False):
        super(TNN, self).__init__()
        self.cnn1 = nn.Conv2d(input_shape[0], 128, (7,7))
        self.cnn2 = nn.Conv2d(128, 256, (5,5))
        self.pooling = nn.MaxPool2d((2,2), (2,2))
        self.CNN_outshape = self._get_conv_output(input_shape)
        self.linear = nn.Linear(self.CNN_outshape, output_size)
             
    def _get_conv_output(self, shape):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        x = self._forward_features(dummy_x)
        CNN_outshape = x.flatten(1).size(1)
        return CNN_outshape
    
    def _forward_features(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pooling(x)
        x = F.relu(self.cnn2(x))
        x = self.pooling(x)
        return x     
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.linear(x.flatten(1))
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
    model = siameseNet()
    summary(model, (16, 1, 160, 160))
    