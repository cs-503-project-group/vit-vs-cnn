import torch.nn as nn
import torch
from timm import models


class ResNet(nn.Module):
    def __init__(self, model_ckpt, ood_threshold):
        super(ResNet, self).__init__()
        resnet = models.resnet50()
        ckpt = torch.load(f"{model_ckpt}")
        resnet.load_state_dict(ckpt) 
        self.resnet = resnet
        self.softmax = nn.Softmax()
        self.ood_threshold = ood_threshold
    
    def forward(self, image):
        logits = self.resnet(image)
        probs = self.softmax(logits)
        
        return probs
