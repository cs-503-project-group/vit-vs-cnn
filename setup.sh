#!/bin/bash 

python generate_synset_ids.py $1 $2
git clone  https://github.com/mf1024/ImageNet-datasets-downloader
cd ImageNet-datasets-downloader
mkdir data
mkdir data/ID
mkdir data/OOD
cd ..
mv ood_download.sh ImageNet-datasets-downloader/
mv id_download.sh ImageNet-datasets-downloader/
pip install -r requirements.txt 
cd ImageNet-datasets-downloader
pip install -r requirements.txt 
chmod 777 ood_download.sh
chmod 777 id_download.sh
./ood_download.sh
./id_download.sh