import torch.nn as nn
import timm


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # resnet = models.resnet50()
        # ckpt = torch.load(f"{model_ckpt}")
        # resnet.load_state_dict(ckpt)
        self.resnet = timm.create_model('resnet50', pretrained=True) 
        self.softmax = nn.Softmax()
        self.name = 'ResNet'
    
    
    def forward(self, image):
        logits = self.resnet(image)
        probs = self.softmax(logits)
        
        return probs
