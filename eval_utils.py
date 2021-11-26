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

            if ind==0:
                probs = batch_probs.cpu().numpy()
                targets = batch_target.cpu().numpy()
            else:
                
                probs = np.concatenate((probs, batch_probs.cpu().numpy()), axis=0)
                targets = np.append(targets, batch_target.cpu().numpy())
      
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))
    f1_scores = np.zeros(len(thresholds))

    for ind,t in enumerate(thresholds):
        maxes = np.max(probs, axis=1)
        preds = maxes < t

        prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', pos_label=1)


        precisions[ind] = prc
        recalls[ind] = rec
        f1_scores[ind] = f1
       

    return precisions, recalls, f1_scores


def evaluate_ID_detection(model, dataloader, thresholds, device, num_classes=1000, batch_size=1):

    probs = torch.zeros((batch_size, num_classes))
    targets = torch.zeros((batch_size, 1))
    model.to(device)
    model.eval()
   
    # check value for threshold
    # prc = Precision(num_classes=num_classes, threshold=1)
    # rec = Recall(num_classes=num_classes, threshold=1)
    # f1  = F1(num_classes=num_classes, threshold=1)

    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0], batch[1]
            inputs.to(device)
            
            batch_probs = model(inputs)
            if ind == 0:
                probs = batch_probs
                targets = batch_target
            else:
                probs = torch.cat((probs, batch_probs), dim=0)
                targets = torch.cat((targets, batch_target), dim=0)
        
    precisions = torch.zeros((len(thresholds), 1))
    recalls    = torch.zeros((len(thresholds), 1))
    f1_scores  = torch.zeros((len(thresholds), 1))

    for ind, t in enumerate(thresholds):
        preds = torch.max(probs, dim=1).index + 1 # the number (index + 1) is the number of the class with maximum probability, which goes from 1 to 1_000
        preds = preds.to(torch.int8)
        
        np_preds = preds.cpu().detach().numpy()
        np_targets = targets.cpu().detach().numpy()

        p, r, f = precision_recall_fscore_support(targets, preds, average='macro')
        precisions[ind] = p
        recalls[ind]    = r
        f1_scores[ind]  = f

    return precisions, recalls, f1_scores
