
from temperature_scaling.temperature_scaling import ModelWithTemperature
from models import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from PIL import Image
from torch import cuda
import os
import timm
import torch



def main():

    device = 'cuda' if cuda.is_available() else 'cpu'
    ori_preprocess = Compose([
        Resize((224), interpolation=Image.BICUBIC),
        CenterCrop(size=(224, 224)),
        ToTensor()])


    imagenet_data = ImageNet(root='../vit-vs-cnn/data/imagenet-val', split='val', transform=ori_preprocess)
    data_loader_ID = DataLoader(imagenet_data, batch_size=16, shuffle=True)

    resnet_model = timm.create_model('resnet50', pretrained=True).to(device)
    deit_model = timm.create_model('deit_small_patch16_224', pretrained=True).to(device)
    mlpmixer_model = timm.create_model('mixer_b16_224', pretrained=True).to(device)
    ecaresnet_model = timm.create_model('ecaresnet50d', pretrained=True).to(device)



    for model, model_name in [(resnet_model, "resnet"), (deit_model, "deit"), (mlpmixer_model, "mlpmixer"), (ecaresnet_model, "ecaresnet")]:
        calibrated_model = ModelWithTemperature(model)

        # Tune the model temperature, and save the results
        calibrated_model.set_temperature(data_loader_ID)
        model_filename =f'../vit-vs-cnn/models/{model_name}_with_temperature.pth'
        torch.save(calibrated_model.state_dict(), model_filename)
        print(f'Temperature scaled {model_name} sved to %s' % model_filename)


    print('Done!')

if __name__ == '__main__':
    main()