import torch.nn as nn
import timm


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # resnet = models.resnet50()
        # ckpt = torch.load(f"{model_ckpt}")
        # resnet.load_state_dict(ckpt)
        self.resnet = timm.create_model('resnet50', pretrained=True) 
        self.softmax = nn.Softmax(dim=1)
        self.name = 'ResNet'
    
    
    def forward(self, image, tmp_scale=False):
        logits = self.resnet(image)

        if tmp_scale:
            probs = self.softmax(logits/1000)
        else:
            probs = self.softmax(logits)
        
        return probs
