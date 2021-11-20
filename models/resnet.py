import torch.nn as nn
import torch
from timm import models


class ResNet(nn.Module):
    def __init__(self, model_ckpt):
        super(ResNet, self).__init__()
        resnet = models.resnet50()
        ckpt = torch.load(f"checkpoints/{model_ckpt}")
        resnet.load_state_dict(ckpt) 
        self.resnet = resnet
        self.softmax = nn.Softmax()
    
    def forward(self, image):
        res = self.resnet(image)
        res = self.softmax(res)
        
        return res