import torch.nn as nn
import timm


class MLPMixer(nn.Module):
    def __init__(self, ood_threshold):
        super(MLPMixer, self).__init__()
        self.mlpmixer = timm.create_model('mixer_s32_224', pretrained=True)
        self.softmax = nn.Softmax()
        self.ood_threshold = ood_threshold
    
    def forward(self, image):
        logits = self.mlpmixer(image)
        probs = self.softmax(logits)
        
        return probs
        
       