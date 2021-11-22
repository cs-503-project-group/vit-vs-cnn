import torch
from torchmetrics import Precision, Recall, F1



def precision(model, dataloader, thresholds, device, num_classes=1000, batch_size=16):

    probs = torch.zeros((batch_size, num_classes))
    targets = torch.zeros((batch_size, 1))
    model.to(device)
    # check value for threshold
    prc = Precision(num_classes = num_classes, threshold=1)

    for ind, input, batch_target in enumerate(dataloader):
        input.to(device)
        batch_probs = model(input)
        if ind==0:
            probs = batch_probs
            targets = batch_target
        else:
            # check
            probs = torch.concat((probs, batch_probs), dim=1)
            targets = torch.concat((targets, batch_target), dim=1)
        
    precisions = torch.zeros((len(thresholds),1))
    for ind,t in enumerate(thresholds):
        # check
        preds = torch.max(probs, dim=1) < t
        precision[ind] = prc(preds, targets)


    return precisions


