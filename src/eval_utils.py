import torch
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

def print_score_recall_f1(model_name, id_prc=None, id_recall=None, id_f1=None, ood_prc=None, ood_recall=None, ood_f1=None, run_id=False, run_ood=False):
    print(f'\n\n--------------- {model_name} ----------------')
    if run_id:
        print(f'ID detection:\n    -Precision: {id_prc} \n    -Score: {id_recall} \n    -F1-score: {id_f1}')
    if run_ood:
        print(f'OOD detection:\n    -Precision: {ood_prc} \n    -Score: {ood_recall} \n    -F1-score: {ood_f1}')


def evaluate_OOD_detection(model, dataloader, thresholds, device, non_semantic=False, tmp_scale=None, use_entropy=False, num_classes=1000):
    
    model_name = model.name
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0].cuda(), batch[1]
            inputs.to(device)
            batch_probs = model(inputs, tmp_scale=tmp_scale)
            if ind == 0:
                probs = batch_probs.cpu().numpy()
                targets = batch_target.cpu().numpy()
            else:
                probs = np.concatenate((probs, batch_probs.cpu().numpy()), axis=0)
                targets = np.append(targets, batch_target.cpu().numpy())

    print('\n-------------------------------')
    print(f'Model: {model.name}')
    
    if use_entropy:
        ID_probs = entropy(probs[targets==1], axis=1)
        OOD_probs = entropy(probs[targets==0], axis=1)
        # larger entropy corresponds to OOD, so pos_label should be 0 (roc_curve assigns pos label to probs higher than the threshold)
        fpr, tpr, ts = roc_curve(y_true=targets, y_score=entropy(probs, axis=1), pos_label=0)
    else:
        ID_probs = np.max(probs[targets==1], axis=1)
        OOD_probs = np.max(probs[targets==0], axis=1)
        fpr, tpr, ts = roc_curve(y_true=targets, y_score=np.max(probs, axis=1))

    OOD_type = "semantic_OOD"
    if non_semantic:
        OOD_type = "non" + OOD_type

    dir_name = f'../vit-vs-cnn/results/{OOD_type}/{use_entropy}_{tmp_scale}/'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    with open(dir_name + f'{model_name}_ID_probs.pickle', 'wb') as f:
            pickle.dump(ID_probs, f)
    with open(dir_name + f'{model_name}_OOD_probs.pickle', 'wb') as f:
            pickle.dump(OOD_probs, f)

    with open(dir_name + f'{model_name}_fpr.pickle', 'wb') as f:
            pickle.dump(fpr, f)
    with open(dir_name + f'{model_name}_tpr.pickle', 'wb') as f:
            pickle.dump(tpr, f)


def evaluate_ID_detection(model, dataloader, device, num_classes=1000):
   
    model_name = model.name
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    target_classes = []
    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0].cuda(), batch[1]
            for t in batch_target:
                if t not in target_classes:
                    target_classes.append(t)
            inputs.to(device) 
            batch_probs = model(inputs)
            if ind == 0:
                probs = batch_probs.cpu().numpy()
                targets = batch_target.cpu().numpy()
            else:
                probs = np.concatenate((probs, batch_probs.cpu().numpy()), axis=0)
                targets = np.append(targets, batch_target.cpu().numpy())
            if ind % 5000 == 0:
                preds = np.argmax(probs, axis=1)
                prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, labels=target_classes, average='macro')
                print_score_recall_f1(model_name=model_name, id_prc=prc, id_recall=rec, id_f1=f1, run_id=True)
   
    preds = np.argmax(probs, axis=1) 
    prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, labels=target_classes, average='macro')
    return prc, rec, f1
