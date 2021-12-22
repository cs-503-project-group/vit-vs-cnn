from models import *
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from PIL import Image
import os
from eval_utils import evaluate_OOD_detection, evaluate_ID_detection
import pickle
from torch import cuda
import json
import matplotlib.pyplot as plt
import pandas as pd


class Image_Dataset(Dataset):
    def __init__(self, id_data_dir, ood_data_dir, transform):
        self.id_data_dir = id_data_dir
        self.ood_data_dir = ood_data_dir
        self.data = []
        self.transform = transform

        # Add ID images to data: [img_name, 0] -- 11 means ID
        folder = os.listdir(self.id_data_dir)[0]
        for file in os.listdir(self.id_data_dir + folder):
            self.data.append((self.id_data_dir + folder + '/' + file, 1))

        # Add OOD images to data: [img_name, 1] -- 0 means OOD
        for folder in os.listdir(self.ood_data_dir):
            for file in os.listdir(self.ood_data_dir + folder):
                self.data.append((self.ood_data_dir + folder + '/' + file, 0))

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        img = None
        with Image.open(img_path).convert('RGB') as im:
            img = self.transform(im)
        return img, target

    def __len__(self):
        return len(self.data)


class MultiDomain_Dataset(Dataset):
    def __init__(self, md_data_dir, transform):
        self.md_data_dir = md_data_dir
        self.data = []
        self.classes = [] # what is this for?
        self.transform = transform
        # Load ground truth labels
        with open('../classes_imagenet/imagenet_r.json') as f_in:
            self.gt_labels = json.load(f_in)

        # Add MD images to data [img_name, gt_label]
        for folder in os.listdir(self.md_data_dir):
            for file in os.listdir(self.md_data_dir + folder):
                if folder+"/"+file in self.gt_labels:
                    gt_label = self.gt_labels[folder+"/"+file]
                    if gt_label not in self.classes:
                        self.classes.append(gt_label)
                    self.data.append((self.md_data_dir + folder + '/' + file, gt_label))

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        with Image.open(img_path).convert('RGB') as im:
            img = self.transform(im) # why pre-process? why this way?

        return img, target

    def __len__(self):
        return len(self.data)


def print_score_recall_f1(model_name, id_prc=None, id_recall=None, id_f1=None, ood_prc=None, ood_recall=None, ood_f1=None, run_id=False, run_ood=False):
    print(f'\n\n--------------- {model_name} ----------------')
    if run_id:
        print(f'ID detection:\n    -Precision: {id_prc} \n    -Recall: {id_recall} \n    -F1-score: {id_f1}')
    if run_ood:
        print(f'OOD detection:\n    -Precision: {ood_prc} \n    -Recall: {ood_recall} \n    -F1-score: {ood_f1}')

def save_to_pickle(list_of_variables_to_save, list_of_variable_names_to_save, non_semantic=True):
    for i, var in enumerate(list_of_variables_to_save):
        file_name = list_of_variable_names_to_save[i]
        if non_semantic:
            file_name = "non-semantic_"+file_name
        with open(f'../results/{file_name}.pickle', 'wb') as f:
            pickle.dump(var, f)

def main(run_ood, run_id, non_semantic, tmp_scale, entropy):
    device = 'cuda' if cuda.is_available() else 'cpu'
    ood_data_dir = '../data/OOD_data/'
    id_data_dir = '../data/ID_data/'
    md_data_dir = '../data/imagenet-r/'

    ori_preprocess = Compose([
            Resize((224), interpolation=Image.BICUBIC),
            CenterCrop(size=(224, 224)),
            ToTensor()])

    # --------- Models
    resnet_model = resnet.ResNet().to(device)
    deit_model = deit.DeiT().to(device)
    mlpmixer_model = mlpmixer.MLPMixer().to(device)
    ecaresnet_model = ecaresnet.ECAResNet().to(device)

    # --------- OOD evaluation
    if run_ood:
        if non_semantic:
            dataset = Image_Dataset(id_data_dir, md_data_dir, transform=ori_preprocess)
        else:
            dataset = Image_Dataset(id_data_dir, ood_data_dir, transform=ori_preprocess)

        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        thresholds = np.arange(0.1, 1.0, step=0.1)

       
        evaluate_OOD_detection(resnet_model,    data_loader, thresholds, device, non_semantic=non_semantic, tmp_scale=tmp_scale, use_entropy=entropy)
        evaluate_OOD_detection(deit_model,      data_loader, thresholds, device, non_semantic=non_semantic, tmp_scale=tmp_scale, use_entropy=entropy)
        evaluate_OOD_detection(mlpmixer_model,  data_loader, thresholds, device, non_semantic=non_semantic, tmp_scale=tmp_scale, use_entropy=entropy)
        evaluate_OOD_detection(ecaresnet_model, data_loader, thresholds, device, non_semantic=non_semantic, tmp_scale=tmp_scale, use_entropy=entropy)
        

    # --------- ID evaluation
    if run_id:

        if non_semantic:
            print('Performing ID evaluation for non semantic shifts')
            imagenet_data = MultiDomain_Dataset(md_data_dir, transform=ori_preprocess)
            data_type = "nonsemantic_OOD" 
        else:
            print('Performing ID evaluation')
            imagenet_data = ImageNet(root='data/imagenet-val', split='val', transform=ori_preprocess)
            data_type = "ID"

        data_loader_ID = DataLoader(imagenet_data, batch_size=16, shuffle=True)

        id_resnet_prc, id_resnet_rec, id_resnet_f1 =          evaluate_ID_detection(resnet_model,    data_loader_ID, device)
        id_deit_prc, id_deit_rec, id_deit_f1 =                evaluate_ID_detection(deit_model,      data_loader_ID, device)
        id_mlpmixer_prc, id_mlpmixer_rec, id_mlpmixer_f1 =    evaluate_ID_detection(mlpmixer_model,  data_loader_ID, device)
        id_ecaresnet_prc, id_ecaresnet_rec, id_ecaresnet_f1 = evaluate_ID_detection(ecaresnet_model, data_loader_ID, device)

        res_dict = {"model": [mlpmixer_model.name, resnet_model.name, ecaresnet_model.name, deit_model.name],
                    "precision": [id_mlpmixer_prc, id_resnet_prc, id_ecaresnet_prc, id_deit_prc],
                    "recall": [id_mlpmixer_rec, id_resnet_rec, id_ecaresnet_rec, id_deit_rec],
                    "f1 score": [id_mlpmixer_f1, id_resnet_f1, id_ecaresnet_f1, id_deit_f1]}

        df = pd.DataFrame(res_dict)
        df.to_csv(f'../results/{data_type}/ID_evaluation.csv')

        print_score_recall_f1('ResNet',    id_resnet_prc,    id_resnet_rec,    id_resnet_f1,    run_id=True)
        print_score_recall_f1('DeiT',      id_deit_prc,      id_deit_rec,      id_deit_f1,      run_id=True)
        print_score_recall_f1('MLPMixer',  id_mlpmixer_prc,  id_mlpmixer_rec,  id_mlpmixer_f1,  run_id=True)
        print_score_recall_f1('ECAResnet', id_ecaresnet_prc, id_ecaresnet_rec, id_ecaresnet_f1, run_id=True)                    

        save_to_pickle([id_resnet_prc,    id_resnet_rec,    id_resnet_f1],    ['id_resnet_prc',    'id_resnet_rec',    'id_resnet_f1'], non_semantic=non_semantic)
        save_to_pickle([id_deit_prc,      id_deit_rec,      id_deit_f1],      ['id_deit_prc',      'id_deit_rec',      'id_deit_f1'], non_semantic=non_semantic)
        save_to_pickle([id_mlpmixer_prc,  id_mlpmixer_rec,  id_mlpmixer_f1],  ['id_mlpmixer_prc',  'id_mlpmixer_rec',  'id_mlpmixer_f1'], non_semantic=non_semantic)
        save_to_pickle([id_ecaresnet_prc, id_ecaresnet_rec, id_ecaresnet_f1], ['id_ecaresnet_prc', 'id_ecaresnet_rec', 'id_ecaresnet_f1'], non_semantic=non_semantic)


if __name__== '__main__':
    main(run_id=False, run_ood=True, non_semantic=True, tmp_scale=False, entropy=False)
