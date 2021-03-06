import torch.nn as nn
import timm


class ECAResNet(nn.Module):
    def __init__(self):
        super(ECAResNet, self).__init__()
        self.ecaresnet = timm.create_model('ecaresnet50d', pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.name = 'ECAResNet'
    
    def forward(self, image, tmp_scale=False):
        logits = self.ecaresnet(image)

        if tmp_scale:
            probs = self.softmax(logits/1000)
        else:
            probs = self.softmax(logits)
        
        return probs
