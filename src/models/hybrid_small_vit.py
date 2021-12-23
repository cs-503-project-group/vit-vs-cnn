import torch.nn as nn
import timm


class Hybrid_S_ViT(nn.Module):
    def __init__(self):
        super(Hybrid_S_ViT, self).__init__()
        
        self.hybrid_vit = timm.create_model('vit_small_r26_s32_224', pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.name = 'Hybrid_Small_ViT'
    
    
    def forward(self, image, tmp_scale=False):
        logits = self.hybrid_vit(image)

        if tmp_scale:
            probs = self.softmax(logits/1000)
        else:
            probs = self.softmax(logits)
        
        return probs
        