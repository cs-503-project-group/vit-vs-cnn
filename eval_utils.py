import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluate_OOD_detection(model, dataloader, thresholds, device, num_classes=1000, batch_size=1):
    probs = np.zeros((batch_size, num_classes))
    targets = np.zeros((batch_size, 1))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0].cuda(), batch[1]
            inputs.to(device)
            batch_probs = model(inputs)
            if ind == 0:
                probs = batch_probs.cpu().numpy()
                targets = batch_target.cpu().numpy()
            else:
                probs = np.concatenate((probs, batch_probs.cpu().numpy()), axis=0)
                targets = np.append(targets, batch_target.cpu().numpy())
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))
    f1_scores = np.zeros(len(thresholds))

    for ind, t in enumerate(thresholds):
        maxes = np.max(probs, axis=1)
        preds = maxes < t
        prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', pos_label=1)
        precisions[ind] = prc
        recalls[ind] = rec
        f1_scores[ind] = f1
    return precisions, recalls, f1_scores


def evaluate_ID_detection(model, dataloader, device, num_classes=1000, batch_size=1):
    probs = np.zeros((batch_size, num_classes))
    targets = np.zeros((batch_size, 1))
    model.to(device)
    model.eval()
    # target_classes = []
    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0].cuda(), batch[1]
            # if batch_target not in target_classes:
            #     target_classes.append(batch_target)
            inputs.to(device) 
            batch_probs = model(inputs)
            if ind == 0:
                probs = batch_probs.cpu().numpy()
                targets = batch_target.cpu().numpy()
            else:
                probs = np.concatenate((probs, batch_probs.cpu().numpy()), axis=0)
                targets = np.append(targets, batch_target.cpu().numpy())
    preds = np.argmax(probs, axis=1) # the number (index + 1) is the number of the class with maximum probability, which goes from 1 to 1_000
    print(preds)
    print(targets)
    # print(f'labels: {classes}')
    target_classes = np.linspace(0, 999)
    prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, labels=target_classes, average='macro')
    return prc, rec, f1
