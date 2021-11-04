# vit-vs-cnn
Project of the course CS-503 "Visual intelligence: Machines &amp; Minds" at EPFL 2021. Performance comparison of ViTs and CNNs on unsemantic distribution shifts (i.e. unseen classes).

## Generate data from specified classes in ImageNet-21k
Here are the steps for downloading data belonging to randomly chosen classes from ImageNet-21k that aren't in ImageNet-1k.
- Clone this repo to your local computer;
- Run `python generate_synset_ids.py 50` to get the synset_ids of 50 random classes from ImageNet-21k that aren't in ImageNet-1k. This command will print the exact command with the random synset_ids that you will need to run in the next step.
- Clone the repo ImageNet-Datasets-Downloader https://github.com/mf1024/ImageNet-datasets-downloader;
- Copy paste the commando that `generate_synset_ids.py` printed, it will download the images from the specified classes. For instance: `python3 ./downloader.py -data_root ./data/ -use_class_list True -class_list n02261063 n10443032 n02602760 n10150071 n01803078 n03019434 n03863783 n03236423 n03538634 n11759404 n10769321 n03850245 n12088909 n10414768 n10052694 n04373563 n12682668 n11610215 n03350204 n01447946 n01731941 n03307037 n10530571 n07730406 n03619650 n12676370 n02408817 n01604330 n12078172 n02177506 n11834890 n13064457 n11816336 n00474881 n02412909 n02016659 n11857696 n04214282 n12251278 n04174705 n11616662 n12862512 n02076402 n10306890 n01788864 n04115256 n10723597 n04292572 n02292085 n13100156 -images_per_class 10` 

The commando above will download the images inside the folder `data`.
