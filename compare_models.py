from models import *
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torchmetrics import Precision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from eval_utils import evaluate_OOD_detection, evaluate_ID_detection
import json
import pickle

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class Image_Dataset(Dataset):
    def __init__(self, id_data_dir, ood_data_dir):
        self.id_data_dir = id_data_dir
        self.ood_data_dir = ood_data_dir
        self.data = []

        for folder in os.listdir(self.id_data_dir)[:10]:
            for file in os.listdir(self.id_data_dir+folder)[:10]:
                self.data.append((self.id_data_dir+folder+"/"+file, 0))

        for folder in os.listdir(self.ood_data_dir)[:1]:
            for file in os.listdir(self.ood_data_dir+folder)[:10]:
                self.data.append((self.ood_data_dir+folder+"/"+file, 1))
        
        

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        img = None
        with Image.open(img_path).convert('RGB') as im:
            img = ori_preprocess(im)

        return img, target


    def __len__(self):
        return len(self.data)


class Image_Dataset_ID(Dataset):
    def __init__(self, id_data_dir):
        self.id_data_dir = id_data_dir
        self.data = []
        self.classes = []
        with open('vit-vs-cnn/classes_imagenet/ground_truth_labels_validation_1k.json') as f_in:
            self.gt_labels = json.load(f_in)
        
        for folder in os.listdir(self.id_data_dir):
            for file in os.listdir(self.id_data_dir+folder)[:10]:
                gt_label = self.gt_labels[file[:-5]]
                if gt_label not in self.classes:
                    self.classes.append(gt_label)

                self.data.append((self.id_data_dir+folder+"/"+file, gt_label))

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        with Image.open(img_path).convert('RGB') as im:
            img = ori_preprocess(im)
        
        return img, target

    def __len__(self):
        return len(self.data)


ood_data_dir = "vit-vs-cnn/data/OOD_data/"
id_data_dir = "vit-vs-cnn/data/ID_data/"

ori_preprocess = Compose([
        Resize((224), interpolation=Image.BICUBIC),
        CenterCrop(size=(224, 224)),
        ToTensor()])

# OOD evaluation
dataset = Image_Dataset(id_data_dir, ood_data_dir)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ID evaluation
dataset_id = Image_Dataset_ID(id_data_dir)
data_loader_ID = DataLoader(dataset_id, batch_size=1, shuffle=True)

thresholds = np.arange(0.5, 0.9, step=0.1)

# --------------------------------------- ResNet ---------------------------------------
resnet_model = resnet.ResNet().to(device)
# the goal is to identify OOD samples
ood_resnet_prc, ood_resnet_rec, ood_resnet_f1 = evaluate_OOD_detection(resnet_model, data_loader, thresholds, device)
id_resnet_prc, id_resnet_rec, id_resnet_f1 = evaluate_ID_detection(resnet_model, data_loader_ID, dataset_id.classes, device)

with open('ood_resnet_prc.pickle', 'wb') as f:
    pickle.dump(ood_resnet_prc, f)
with open('ood_resnet_rec.pickle', 'wb') as f:
    pickle.dump(ood_resnet_rec, f)
with open('ood_resnet_f1.pickle', 'wb') as f:
    pickle.dump(ood_resnet_f1, f)

with open('id_resnet_prc.pickle', 'wb') as f:
    pickle.dump(id_resnet_prc, f)
with open('id_resnet_rec.pickle', 'wb') as f:
    pickle.dump(id_resnet_rec, f)
with open('id_resnet_f1.pickle', 'wb') as f:
    pickle.dump(id_resnet_f1, f)

print(id_resnet_prc, id_resnet_rec, id_resnet_f1)
print(ood_resnet_prc, ood_resnet_rec, ood_resnet_f1)



# --------------------------------------- DeiT ---------------------------------------
deit_model = deit.DeiT().to(device)

ood_deit_prc, ood_deit_rec, ood_deit_f1 = evaluate_OOD_detection(deit_model, data_loader, thresholds, device)
id_deit_prc, id_deit_rec, id_deit_f1 = evaluate_ID_detection(deit_model, data_loader_ID, dataset_id.classes, device)

with open('ood_deit_prc.pickle', 'wb') as f:
    pickle.dump(ood_deit_prc, f)
with open('ood_deit_rec.pickle', 'wb') as f:
    pickle.dump(ood_deit_rec, f)
with open('ood_deit_f1.pickle', 'wb') as f:
    pickle.dump(ood_deit_f1, f)

with open('id_deit_prc.pickle', 'wb') as f:
    pickle.dump(id_deit_prc, f)
with open('id_deit_rec.pickle', 'wb') as f:
    pickle.dump(id_deit_rec, f)
with open('id_deit_f1.pickle', 'wb') as f:
    pickle.dump(id_deit_f1, f)

print(id_deit_prc, id_deit_rec, id_deit_f1)
print(ood_deit_prc, ood_deit_rec, ood_deit_f1)


# ConvMixer
# convmixer_model = convmixer.ConvMixer().to(device)

# convmixer_prc, convmixer_rec, convmixer_f1 = evaluate_OOD_detection(convmixer_model, data_loader, thresholds, device)
# print(convmixer_prc, convmixer_rec, convmixer_f1)


# --------------------------------------- MLPMixer ---------------------------------------
mlpmixer_model = mlpmixer.MLPMixer().to(device)

ood_mlpmixer_prc, ood_mlpmixer_rec, ood_mlpmixer_f1 = evaluate_OOD_detection(mlpmixer_model, data_loader, thresholds, device)
id_mlpmixer_prc, id_mlpmixer_rec, id_mlpmixer_f1 = evaluate_ID_detection(mlpmixer_model, data_loader_ID, dataset_id.classes, device)

with open('ood_mlpmixer_prc.pickle', 'wb') as f:
    pickle.dump(ood_mlpmixer_prc, f)
with open('ood_mlpmixer_rec.pickle', 'wb') as f:
    pickle.dump(ood_mlpmixer_rec, f)
with open('ood_mlpmixer_f1.pickle', 'wb') as f:
    pickle.dump(ood_mlpmixer_f1, f)

with open('id_mlpmixer_prc.pickle', 'wb') as f:
    pickle.dump(id_mlpmixer_prc, f)
with open('id_mlpmixer_rec.pickle', 'wb') as f:
    pickle.dump(id_mlpmixer_rec, f)
with open('id_mlpmixer_f1.pickle', 'wb') as f:
    pickle.dump(id_mlpmixer_f1, f)

print(id_mlpmixer_prc, id_mlpmixer_rec, id_mlpmixer_f1)
print(ood_mlpmixer_prc, ood_mlpmixer_rec, ood_mlpmixer_f1)

# --------------------------------------- EcaResNet ---------------------------------------
ecaresnet_model = ecaresnet.ECAResNet().to(device)

ood_ecaresnet_prc, ood_ecaresnet_rec, ood_ecaresnet_f1 = evaluate_OOD_detection(ecaresnet_model, data_loader, thresholds, device)
id_ecaresnet_prc, id_ecaresnet_rec, id_ecaresnet_f1 = evaluate_ID_detection(ecaresnet_model, data_loader_ID, dataset_id.classes, device)


with open('ood_ecaresnet_prc.pickle', 'wb') as f:
    pickle.dump(ood_ecaresnet_prc, f)
with open('ood_ecaresnet_rec.pickle', 'wb') as f:
    pickle.dump(ood_ecaresnet_rec, f)
with open('ood_ecaresnet_f1.pickle', 'wb') as f:
    pickle.dump(ood_ecaresnet_f1, f)

with open('id_ecaresnet_prc.pickle', 'wb') as f:
    pickle.dump(id_ecaresnet_prc, f)
with open('id_ecaresnet_rec.pickle', 'wb') as f:
    pickle.dump(id_ecaresnet_rec, f)
with open('id_ecaresnet_f1.pickle', 'wb') as f:
    pickle.dump(id_ecaresnet_f1, f)

print(id_ecaresnet_prc, id_ecaresnet_rec, id_ecaresnet_f1)
print(ood_ecaresnet_prc, ood_ecaresnet_rec, ood_ecaresnet_f1)