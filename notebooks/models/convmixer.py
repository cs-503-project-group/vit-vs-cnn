import torch.nn as nn
import timm


class ConvMixer(nn.Module):
    def __init__(self, ood_threshold):
        super(ConvMixer, self).__init__()
        self.convmixer = timm.create_model('convmixer_1024_20_ks9_p14', pretrained=True)
        self.softmax = nn.Softmax()
    
    def forward(self, image):
        logits = self.convmixer(image)
        probs = self.softmax(logits)
        
        return probs