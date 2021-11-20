import torch.nn as nn
import timm


class MLPMixer(nn.Module):
    def __init__(self):
        super(MLPMixer, self).__init__()
        self.mlpmixer = timm.create_model('mixer_s32_224', pretrained=True)
        self.softmax = nn.Softmax()
    
    def forward(self, image):
        res = self.mlpmixer(image)
        res = self.softmax(res)
        
        return res
    