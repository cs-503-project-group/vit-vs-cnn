import torch.nn as nn
import timm


class MLPMixer(nn.Module):
    def __init__(self):
        super(MLPMixer, self).__init__()
        
        self.mlpmixer = timm.create_model('mixer_b16_224', pretrained=True)
        self.softmax = nn.Softmax()
    
    
    def forward(self, image):
        logits = self.mlpmixer(image)
        probs = self.softmax(logits)
        
        return probs
        
       