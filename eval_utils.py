import torch
from torchmetrics import Precision, Recall, F1



def evaluate_OOD_detection(model, dataloader, thresholds, device, num_classes=1000, batch_size=1):

    probs = torch.zeros((batch_size, num_classes))
    targets = torch.zeros((batch_size, 1))
    model.to(device)
    model.eval()
   
    # check value for threshold
    prc = Precision(num_classes = 2, threshold=1)
    rec = Recall(num_classes = 2, threshold=1)
    f1 = F1(num_classes = 2, threshold=1)
    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0], batch[1]
            inputs.to(device)
            
            batch_probs = model(inputs)
            if ind==0:
                probs = batch_probs
                targets = batch_target
            else:
                
                probs = torch.cat((probs, batch_probs), dim=0)
                targets = torch.cat((targets, batch_target), dim=0)
        
    precisions = torch.zeros((len(thresholds),1))
    recalls = torch.zeros((len(thresholds),1))
    f1_scores = torch.zeros((len(thresholds),1))

    for ind,t in enumerate(thresholds):
        maxes = torch.max(probs, dim=1).values
        preds = maxes < t
        preds = preds.to(torch.int8)
        
        precisions[ind] = prc(preds, targets)
        recalls[ind] = rec(preds, targets)
        f1_scores[ind] = f1(preds, targets)
       

    return precisions, recalls, f1_scores


