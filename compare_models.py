from models import *
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torchmetrics import Precision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from eval_utils import evaluate_OOD_detection
import json


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class Image_Dataset(Dataset):
    def __init__(self, id_data_dir, ood_data_dir):
        self.id_data_dir = id_data_dir
        self.ood_data_dir = ood_data_dir
        self.data = []

        for file in os.listdir(self.id_data_dir)[:10]:
            self.data.append((self.id_data_dir+file, 0))

        for folder in os.listdir(self.ood_data_dir)[:1]:
            for file in os.listdir(self.ood_data_dir+folder)[:10]:
                self.data.append((self.ood_data_dir+f"{folder}/"+file, 1))

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        img = ori_preprocess(Image.open(img_path))
        return img, target

    def __len__(self):
        return len(self.data)


class Image_Dataset_ID(Dataset):
    def __init__(self, id_data_dir):
        self.id_data_dir = id_data_dir
        self.data = []
        with open('classes_imagenet/ground_truth_labels_validation_1k.json') as f_in:
            self.gt_labels = json.load(f_in)
        
        for file in os.listdir(self.id_data_dir):
            gt_label = self.gt_labels[file]
            self.data.append((self.id_data_dir+file, gt_label))

    def __get__item__(self, idx):
        img_path, target = self.data[idx]
        img = ori_preprocess(Image.open(img_path))
        return img, target

    def __len__(self):
        return len(self.data)





ood_data_dir = "/home/SHARED_FOLDER/data/OOD_data/"
id_data_dir = "/home/SHARED_FOLDER/data/ID_data/imagenet1k-val/"

ori_preprocess = Compose([
        Resize((224), interpolation=Image.BICUBIC),
        CenterCrop(size=(224, 224)),
        ToTensor()])

# OOD evaluation
dataset = Image_Dataset(id_data_dir, ood_data_dir)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ID evaluation
dataset_id = Image_Dataset_ID(id_data_dir, ood_data_dir)
data_loader_ID = DataLoader(dataset, batch_size=1, shuffle=True)

thresholds = torch.range(0.5, 0.9, step=0.1)

# --------------------------------------- ResNet ---------------------------------------
ood_threshold = 0.5
resnet_model = resnet.ResNet().to(device)
# the goal is to identify OOD samples

resnet_prc, resnet_rec, resnet_f1 = evaluate_OOD_detection(resnet_model, data_loader, thresholds, device)
print(resnet_prc, resnet_rec, resnet_f1)



# --------------------------------------- DeiT ---------------------------------------
deit_model = deit.DeiT().to(device)

deit_prc, deit_rec, deit_f1 = evaluate_OOD_detection(deit_model, data_loader, thresholds, device)
print(deit_prc, deit_rec, deit_f1)


# --------------------------------------- ConvMixer ---------------------------------------
convmixer_model = convmixer.ConvMixer().to(device)

convmixer_prc, convmixer_rec, convmixer_f1 = evaluate_OOD_detection(convmixer_model, data_loader, thresholds, device)
print(convmixer_prc, convmixer_rec, convmixer_f1)


# --------------------------------------- MLPMixer ---------------------------------------
mlpmixer_model = mlpmixer.MLPMixer().to(device)

mlpmixer_prc, mlpmixer_rec, mlpmixer_f1 = evaluate_OOD_detection(mlpmixer_model, data_loader, thresholds, device)
print(mlpmixer_prc, mlpmixer_rec, mlpmixer_f1)

# --------------------------------------- EcaResNet ---------------------------------------
ecaresnet_model = ecaresnet.ECAResNet().to(device)
ecaresnet_prc, ecaresnet_rec, ecaresnet_f1 = evaluate_OOD_detection(ecaresnet_model, data_loader, thresholds, device)
print(ecaresnet_prc, ecaresnet_rec, ecaresnet_f1)