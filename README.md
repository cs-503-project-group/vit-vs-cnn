# vit-vs-cnn
Project of the course CS-503 "Visual intelligence: Machines &amp; Minds" at EPFL 2021. Performance comparison of ViTs and CNNs on unsemantic distribution shifts (i.e. unseen classes).

<!-- ## Generate data from specified classes in ImageNet-21k with setup.sh script
- run `setup.sh n`, where n are OOD classes. It should download all nessesary data and install all requirements. It is advised to use virtual environment before
(`pip install virtualenv`, `virtualenv vi`, `source vi/bin/activate` )   -->

## Download the data

### Download OOD data folder from EPFL google drive

Best solution for downloading OOD data would be download the already prepered data from [google drive](https://drive.google.com/drive/folders/14PcKqll5M7_byC1G85fn0O1wuydYNG7J?usp=sharing). Inside there are OOD data. Download the `OOD_data` folder and put it inside project in ```data```. For `ID data`, `Validation set` and `Non Semantic OoD Data` follow the instructions below.

###  Or instead of downloading OOD data, generate data from specified classes in ImageNet-21k 

Here are the steps for downloading data belonging to randomly chosen classes from ImageNet-21k that aren't in ImageNet-1k.
- Clone this repo to your local computer;
- Run `python generate_synset_ids.py` to get the synset_ids of n random classes from ImageNet-21k that aren't in ImageNet-1k. This command will the exact command with the random synset_ids that you will need to run in the next step.
- Clone the repo ImageNet-Datasets-Downloader https://github.com/mf1024/ImageNet-datasets-downloader;
- Create a folder called `/data` inside the repo ImageNet-Datasets-Downloader. The images downloaded will end up there.
- Copy paste the commando that `generate_synset_ids.py` printed, it will download the images from the specified classes. For instance: `python3 ./downloader.py -data_root ./data/ -use_class_list True -class_list n02261063 n10443032 n02602760 n10150071 n01803078 n03019434 n03863783 n03236423 n03538634 n11759404 n10769321 n03850245 n12088909 n10414768 n10052694 n04373563 n12682668 n11610215 n03350204 n01447946 n01731941 n03307037 n10530571 n07730406 n03619650 n12676370 n02408817 n01604330 n12078172 n02177506 n11834890 n13064457 n11816336 n00474881 n02412909 n02016659 n11857696 n04214282 n12251278 n04174705 n11616662 n12862512 n02076402 n10306890 n01788864 n04115256 n10723597 n04292572 n02292085 n13100156 -images_per_class 10` 

## Non Semantic OoD Data

You can download the ImageNet-R dataset from [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar). Then untar the downloaded folder and put it in ```data/``` directory.

## Validation set
Go to <https://image-net.org/challenges/LSVRC/2012/2012-downloads.php> download `Development Kit tasks 1 & 2` and `validation images`. Put them in ```data/imagenet-val``` folder. 

## ID data

Untar `validation images` from previous point to to the ```data/ID_data``` folder

## Project structure

- scripts used for helping with data generation are in ```src/data_generation``` folder
- models used for this project are in ```src/models```
- Comparison between models is in ```src/compare_models.py``` where by changing main parameters you can setup experiment with ID detection, OoD detection, temprecure scaling or using entropy.
- plots generated for raport are in ```src/plot.py``` script

## Usage
1. Install requierments from `requirements.txt`
2. Download the data as written above
3. Run experiment by running `src/compare_models.py` there are five arguments which will describe the experiment.
  - `run_id`: if set, the image classification experiment will be performed.
  - `run_ood`: if set, the OoD detection experiment will be performed.
  - `non_semantic`: if set, imagenet-r will be used as the OoD data, otherwise our curated OoD dataset (semantic shifts) will be used.
  - `tmp_scale`: if set, softmax with temperature scaling will be used.
  - `entropy`: if set, the entropy of softmax probability distributions will be used, otherwise the maximum softmax probability.
  
This will run the experiments and store the results in as pickle files in their corresponding folder in `results` directory. The pickle files can then be used to generate plots as explained below.

4. Optionally run `src/plot.py` for generating the results plots of previously daved models. You can indicate the type of experiment using `--nonsemantic`, `--entropy`, or `--temp_scale`. This will generate plots and store them in the directory dedicated to the experiment in `results` folder. (Our generated plot are laready there :) )
5. For example for generating plots corresponding to experiment compating models over non-semantic shifts using entropy of softmax run the following command:

```
python plot.py --nonsemantic --entropy
```

And to generate plots corresponding to experiment compating models over semantic shifts using tempearture scaling run the following command:

``` 
python plot.py --temp_scale
```
