import torch.nn as nn
import timm


class MLPMixer(nn.Module):
    def __init__(self):
        super(MLPMixer, self).__init__()
        
        self.mlpmixer = timm.create_model('mixer_b16_224', pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.name = 'MLPMixer'
    
    
    def forward(self, image, tmp_scale=False):
        logits = self.mlpmixer(image)

        if tmp_scale:
            probs = self.softmax(logits/1000)
        else:
            probs = self.softmax(logits)
        
        return probs
        
       
