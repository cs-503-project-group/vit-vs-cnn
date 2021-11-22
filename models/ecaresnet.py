import torch.nn as nn
import timm


class ECAResNet(nn.Module):
    def __init__(self):
        super(ECAResNet, self).__init__()
        self.ecaresnet = timm.create_model('ecaresnet50d', pretrained=True)
        self.softmax = nn.Softmax()
    
    def forward(self, image):
        logits = self.ecaresnet(image)
        probs = self.softmax(logits)
        
        return probs