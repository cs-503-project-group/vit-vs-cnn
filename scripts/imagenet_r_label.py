import os
import pandas as pd

ImagenetR_path = "vit-vs-cnn/data/imagenet-r/"
csv_file_path = "vit-vs-cnn/classes_imagenet/imagenet_r.csv"

data_dict = {}

categories = open('vit-vs-cnn/classes_imagenet/classes_in_imagenet_1k.txt', 'r')
indices = open('vit-vs-cnn/classes_imagenet/imagenet1000_clsidx_to_labels.txt', 'r')
label2idx = {}
cat2idx = {}
categories = categories.readlines()
indices  = indices.readlines()    


for i,line in enumerate(indices):
    line = line.split(":")
    idx = int(line[0])
    label = line[1][2:-1]
    if i<len(indices)-1: 
        label = label[:-2]
    label2idx[label] = idx


count = 0
for i,line in enumerate(categories):
    line = line.split(":")
    cat = line[0]
    label = line[1][1:]  
    if i<len(categories)-1: 
        label = label[:-1] 
    cat2idx[cat] = label2idx[label]
    count+=1


print(cat2idx)
data = {}
for folder in os.listdir(ImagenetR_path):
    for file in os.listdir(ImagenetR_path+folder):
        data[file] = cat2idx[folder]



data_frame = pd.Dataframe(data)
data_frame.to_csv(csv_file_path)

