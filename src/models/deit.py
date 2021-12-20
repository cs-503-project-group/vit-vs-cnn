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
        self.softmax = nn.Softmax(dim=1)
        self.name = 'DeiT'
        
    
    def forward(self, image, tmp_scale=None):
        logits = self.deit(image)

        if tmp_scale:
            probs = self.softmax(logits/tmp_scale)
        else:
            probs = self.softmax(logits)
        
        return probs