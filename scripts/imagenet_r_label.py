import os
import pandas as pd
from pathlib import Path
import json

ImagenetR_path = "../vit-vs-cnn/data/imagenet-r/"
json_file_path = Path("../vit-vs-cnn/classes_imagenet/imagenet_r.json")

data_dict = {}

categories = open('../vit-vs-cnn/classes_imagenet/classes_in_imagenet_1k.txt', 'r')
indices = open('../vit-vs-cnn/classes_imagenet/imagenet1000_clsidx_to_labels.txt', 'r')
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



data = {}
for folder in os.listdir(ImagenetR_path):
    for file_ in os.listdir(ImagenetR_path+folder):
        if folder in cat2idx:
            data[folder+"/"+file_] = cat2idx[folder]

with open(json_file_path, 'w') as fp:
    json.dump(data, fp)

# print(data)
# data_frame = pd.DataFrame.from_dict(data, orient='index', columns= ["label"])
# data_frame.to_csv(csv_file_path)

