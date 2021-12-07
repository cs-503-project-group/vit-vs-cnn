import matplotlib.pyplot as plt 
import numpy as np
import pickle


def read_metric(file_name):
    objects = None
    with (open(file_name, "rb")) as openfile:
      
        try:
            objects = pickle.load(openfile)
        except EOFError:
            print("error")

    return objects


ood_resnet_prc = read_metric('ood_resnet_prc.pickle')
ood_resnet_rec = read_metric('ood_resnet_rec.pickle')
ood_resnet_f1 = read_metric('ood_resnet_f1.pickle')



ood_deit_prc = read_metric('ood_deit_prc.pickle')
ood_deit_rec = read_metric('ood_deit_rec.pickle')
ood_deit_f1 = read_metric('ood_deit_f1.pickle')


ood_mlpmixer_prc = read_metric('ood_mlpmixer_prc.pickle')
ood_mlpmixer_rec = read_metric('ood_mlpmixer_rec.pickle')
ood_mlpmixer_f1 = read_metric('ood_mlpmixer_f1.pickle')


ood_ecaresnet_prc = read_metric('ood_ecaresnet_prc.pickle')
ood_ecaresnet_rec = read_metric('ood_ecaresnet_rec.pickle')
ood_ecaresnet_f1 = read_metric('ood_ecaresnet_f1.pickle')

# data to be plotted
x = np.arange(0.1, 0.5, step=0.05)
 
# plotting
# plt.title("OOD Detection Precision")
# plt.xlabel("OOD threshold")
# plt.ylabel("Precision")
# plt.plot(x, ood_resnet_prc, label="ResNet50")
# plt.plot(x, ood_deit_prc, label="DeiT")
# plt.plot(x, ood_mlpmixer_prc, label="MLPMixer")
# plt.plot(x, ood_ecaresnet_prc, label="EcaResNet")
# plt.legend()
# plt.show()
# plt.savefig("precision.jpg")

# plt.title("OOD Detection Recall")
# plt.xlabel("OOD threshold")
# plt.ylabel("Recall")
# plt.plot(x, ood_resnet_rec, label="ResNet50")
# plt.plot(x, ood_deit_rec, label="DeiT")
# plt.plot(x, ood_mlpmixer_rec, label="MLPMixer")
# plt.plot(x, ood_ecaresnet_rec, label="EcaResNet")
# plt.legend()
# plt.show()
# plt.savefig("recall.jpg")

plt.title("OOD Detection F1-Score")
plt.xlabel("OOD threshold")
plt.ylabel("F1-Score")
plt.plot(x, ood_resnet_f1, label="ResNet50")
plt.plot(x, ood_deit_f1, label="DeiT")
plt.plot(x, ood_mlpmixer_f1, label="MLPMixer")
plt.plot(x, ood_ecaresnet_f1, label="EcaResNet")
plt.legend()
plt.show()
plt.savefig("f1.jpg")