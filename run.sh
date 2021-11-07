#!/bin/bash 

python generate_synset_ids.py $1 $2
git clone  https://github.com/mf1024/ImageNet-datasets-downloader
cd ImageNet-Datasets-Downloader
mkdir data
mkdir data/ID
mkdir data/OOD
cd ..
mv ood_download.sh ImageNet-Datasets-Downloader/
mv id_download.sh ImageNet-Datasets-Downloader/
pip install -r requirements.txt 
cd ImageNet-Datasets-Downloader
pip install -r requirements.txt 
./ood_download.sh
./id_download.sh