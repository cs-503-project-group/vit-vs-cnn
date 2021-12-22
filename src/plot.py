import matplotlib.pyplot as plt 
import numpy as np
import pickle
import argparse
from sklearn.metrics import auc

def read_pickle(file_name):
    objects = None
    with (open(file_name, "rb")) as openfile:
      
        try:
            objects = pickle.load(openfile)
        except EOFError:
            print("error")

    return objects

def box_plot(model_names, dir_name):
    
    for model_name in model_names:
        OOD_probs = read_pickle(dir_name + f'{model_name}_OOD_probs.pickle')
        ID_probs = read_pickle(dir_name + f'{model_name}_ID_probs.pickle')
    
        print(f"{model_name}\n mean diff:{np.mean(ID_probs)-np.mean(OOD_probs)}")
        plt.boxplot([ID_probs, OOD_probs], showmeans=True, labels=["ID Data", "OOD Data"])
        plt.title(model_name, fontsize=15)
        plt.show()
        plt.savefig(fname=f"{dir_name}{model_name}_box_plot")
        plt.clf()

def per_model_roc_cruve(model_names, dir_name):
   
    for model_name in model_names:
        fpr = read_pickle(dir_name + f'{model_name}_fpr.pickle')
        tpr = read_pickle(dir_name + f'{model_name}_tpr.pickle')

        roc_auc = auc(x=fpr, y=tpr)

        plt.plot(fpr, tpr, 'b')
        plt.plot([0, 1], [0, 1],'k--')
        plt.plot([0, 1], [0.9, 0.9], 'g--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(f'{model_name} FPR when TPR is {tpr[tpr>=0.9][0]}: {fpr[tpr>=0.9][0]}', fontsize=15)
        plt.savefig(fname=f"{dir_name}{model_name}_roc_curve")
        plt.clf()


def roc_curve(model_names, dir_name, tmp_scale=False, entropy=False):

    if tmp_scale:
        caption = "ROC-temperature scaling"
    elif entropy:
        caption = "ROC-entropy"
    else:
        caption = "ROC-raw softmax"
        
    plt.title(caption, fontsize=15)

    for model_name in model_names:
        fpr = read_pickle(dir_name + f'{model_name}_fpr.pickle')
        tpr = read_pickle(dir_name + f'{model_name}_tpr.pickle')

        roc_auc = auc(x=fpr, y=tpr)

        plt.plot(fpr, tpr, label = f'{model_name} AUC = %0.2f' % roc_auc)

    plt.plot([0, 1], [0, 1],'k--')
    plt.plot([0, 1], [0.9, 0.9], 'g--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.legend(loc = 'lower right', fontsize=14)
    plt.savefig(fname=f"{dir_name}roc_curve")
    

def main():
    # add arguments: entropy, temp_scale, non_semantic
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp_scale", type=int, 
                        help="the temperature used in temperature scaling. If not set, no temperature scaling will be used.")

    parser.add_argument("--entropy", action="store_true",
                        help="if set, the prediction is performed based on entopy of softmax probabiities, and max probability otherwise.")
    
    parser.add_argument("--nonsemantic", action="store_true",
                        help="if set, the evaluation is performed for non-semantic distribution shifts.")

    args = parser.parse_args()
    OOD_type = "semantic_OOD"
    if args.nonsemantic:
        OOD_type = "non" + OOD_type
    # plot box_plots and roc curves
    # TODO: the dir_name might need adjustments based on your root directory
    dir_name = f'../vit-vs-cnn/results/{OOD_type}/'
    if args.entropy:
        dir_name += 'entropy/'
    elif args.temp_scale:
        dir_name += 'tmp_scale/'
    else:
        dir_name += 'softmax/'
        
    model_names = ["ResNet", "MLPMixer", "ECAResNet", "DeiT"]

    box_plot(model_names, dir_name)
    roc_curve(model_names, dir_name, args.temp_scale, args.entropy)
    per_model_roc_cruve(model_names, dir_name)


    
if __name__== '__main__':
    main()
