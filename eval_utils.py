import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
# from compare_models import print_score_recall_f1

def print_score_recall_f1(model_name, id_prc=None, id_recall=None, id_f1=None, ood_prc=None, ood_recall=None, ood_f1=None, run_id=False, run_ood=False):
    print(f'\n\n--------------- {model_name} ----------------')
    if run_id:
        print(f'ID detection:\n    -Precision: {id_prc} \n    -Score: {id_recall} \n    -F1-score: {id_f1}')
    if run_ood:
        print(f'OOD detection:\n    -Precision: {ood_prc} \n    -Score: {ood_recall} \n    -F1-score: {ood_f1}')


def evaluate_OOD_detection(model, dataloader, thresholds, device, color, num_classes=1000, batch_size=8):
    probs = np.zeros((batch_size, num_classes))
    targets = np.zeros((batch_size, 1))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            inputs, batch_target = batch[0].cuda(), batch[1]
            inputs.to(device)
            batch_probs = model(inputs) # shape: (1, 1000)
            if ind == 0:
                probs = batch_probs.cpu().numpy()
                targets = batch_target.cpu().numpy()
            else:
                probs = np.concatenate((probs, batch_probs.cpu().numpy()), axis=0)
                targets = np.append(targets, batch_target.cpu().numpy())

    print('\n-------------------------------')
    print(f'Model: {model.name}')
    
    fpr, tpr, ts = roc_curve(y_true=targets, y_score=np.max(probs, axis=1))
    roc_auc = auc(x=fpr, y=tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = f'{model.name} AUC = %0.2f' % roc_auc, color=color)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig(fname=f'../vit-vs-cnn/pickles/roc_for_{model.name}')

    # precisions = np.zeros(len(thresholds))
    # recalls = np.zeros(len(thresholds))
    # f1_scores = np.zeros(len(thresholds))

    # # for ind, t in enumerate(thresholds):
    # #     maxes = np.max(probs, axis=1)
    # #     preds = maxes > t # if the max is greater than the threshold: predict it as 1 (=in-distribution)
    # #     prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', pos_label=1)
    # #     precisions[ind] = prc
    # #     recalls[ind] = rec
    # #     f1_scores[ind] = f1
    
    # return precisions, recalls, f1_scores


def evaluate_ID_detection(model, dataloader, device, num_classes=1000, batch_size=8):
    probs = np.zeros((batch_size, num_classes))
    targets = np.zeros((batch_size, 1))
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
    print(preds)
    print(targets)
    prc, rec, f1, _ = precision_recall_fscore_support(targets, preds, labels=target_classes, average='macro')
    return prc, rec, f1
