import torch.nn as nn
import torch


class DeiT(nn.Module):
    def __init__(self, model_ckpt, ood_threshold):
        super(DeiT, self).__init__()
        deit =  torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224')
        ckpt = torch.load(f"checkpoints/{model_ckpt}")
        deit.load_state_dict(ckpt) 
        self.deit = deit
        self.softmax = nn.Softmax()
        self.ood_threshold = ood_threshold
    
    def forward(self, image):
        logits = self.deit(image)
        probs = self.softmax(logits)
        
        return probs