from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size=14):
        super(Encoder, self).__init__()

        self.enc_img_size = embed_size

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((embed_size, embed_size))

    def forward(self, images):
        with torch.no_grad():
            out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)

        return out