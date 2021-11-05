#!/usr/bin/env python3
import csv
import random
import sys

# Read synset ids of classes in ImageNet-21k
synset_ids_21k = []
with open('classes_in_imagenet_21k.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        synset_ids_21k.append(row[0].split(',')[0])
#print('Done creating synset_ids_21k')

# Read synset ids of classes in ImageNet-1k
synset_ids_1k = []
with open('classes_in_imagenet_1k.txt') as txtfile:
    lines = txtfile.readlines()
    for line in lines:
        synset_ids_1k.append(line.split(':')[0])
#print('Done creating synset_ids_1k')

# Generate random synset_ids from synset_ids_21k which are not in synset_ids_1k
nr_ood_classes = int(sys.argv[1])
ood_synset_ids = []
ood_synset_ids_str = ''
max_id = len(synset_ids_21k)
i = 0

while i < nr_ood_classes:
    random_synset_id = synset_ids_21k[random.randint(0, max_id)]
    if random_synset_id not in synset_ids_1k:
        #synset_ids.append(random_synset_id)
        ood_synset_ids_str += random_synset_id + ' '
        i += 1

nr_id_classes = int(sys.argv[2])
id_synset_ids = []
id_synset_ids_str = ''
max_id = len(synset_ids_1k)
i = 0

while i < nr_id_classes:
    random_synset_id = synset_ids_1k[random.randint(0, max_id)]
        #synset_ids.append(random_synset_id)
    id_synset_ids_str += random_synset_id + ' '
    i += 1
# Print the commando to run in ImageNet-Dataset-Downloader

# print('Random synset ids: {}'.format(synset_ids_str))
print('python ./downloader.py -data_root ./data/OOD/ -use_class_list True -class_list {} -images_per_class 10'.format(ood_synset_ids_str))
print('python ./downloader.py -data_root ./data/ID/ -use_class_list True -class_list {} -images_per_class 10'.format(id_synset_ids_str))




