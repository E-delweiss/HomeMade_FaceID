import torch
import torch.nn as nn
import torchvision

class SiameseNet(nn.Module):
    def __init__(self):
        """Custom the FaceNet InceptionResnetV1 model.
        """
        super(SiameseNet, self).__init__()
        self.FaceNetModif_seq1 = nn.Sequential(*list(InceptionResnetV1(pretrained='vggface2').children())[:-3])
        self.FaceNetModif_seq2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792, 512, bias=True),
            nn.Linear(512, 256, bias=True),
            nn.Linear(256,128, bias=True),
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
