import torch.nn as nn
import torch
import timm

class DeiT(nn.Module):
    def __init__(self):
        super(DeiT, self).__init__()
        # deit =  torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224')
        # ckpt = torch.load(f"checkpoints/{model_ckpt}")
        # deit.load_state_dict(ckpt) 
        self.deit = timm.create_model('deit_small_patch16_224', pretrained=True)
        self.softmax = nn.Softmax()
        
    
    def forward(self, image):
        logits = self.deit(image)
        probs = self.softmax(logits)
        
        return probs