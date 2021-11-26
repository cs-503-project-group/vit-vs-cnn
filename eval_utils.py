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


