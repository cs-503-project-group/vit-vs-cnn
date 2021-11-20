import torch.nn as nn
import torch


class DeiT(nn.Module):
    def __init__(self, model_ckpt):
        super(DeiT, self).__init__()
        deit =  torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224')
        ckpt = torch.load(f"checkpoints/{model_ckpt}")
        deit.load_state_dict(ckpt) 
        self.deit = deit
        self.softmax = nn.Softmax()
    
    def forward(self, image):
        res = self.deit(image)
        res = self.softmax(res)
        
        return res