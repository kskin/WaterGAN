# WaterGAN

<p align="center">
  <img src="https://github.com/kskin/WaterGAN/blob/master/watergan.PNG?raw=true"/>
</p>

+ This repository contains source code for WaterGAN developed in [WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images](https://arxiv.org/abs/1702.07392).
+ This code is modified from [Taehoon Kim's](http://carpedm20.github.io/)
  [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) (MIT-licensed). Our modifications are [MIT-licensed](./LICENSE).

# Usage

Create a data directory and download data:
1) MHL test tank dataset:  [MHL.tar.gz](http://www.umich.edu/~dropopen/MHL.tar.gz)
2) Jamaica field dataset: [Jamaica.tar.gz](http://www.umich.edu/~dropda/Jamaica.tar.gz)

Train a model:

```
python mainmhl.py --water_dataset water_images --air_dataset air_images --depth_dataset air_depth
```

# Color Correction Network

WaterGAN outputs a dataset with paired true color, depth, and (synthetic) underwater images. We can use this to train an end-to-end network for underwater image restoration. Source code and pretrained models for the end-to-end network are available [here](https://github.com/ljlijie/WaterGAN-color-correction-net). For more details, see the [paper](https://arxiv.org/abs/1702.07392).
  
# Citations

If you find this work useful for your research please cite WaterGAN in your publications.

```
@article{li2017watergan,
    author    = {Jie Li, Katherine A. Skinner, Ryan M. Eustice and
               Matthew Johnson{-}Roberson},
  title     = {WaterGAN: Unsupervised Generative Network to Enable Real-time Color
               Correction of Monocular Underwater Images},
  journal   = {CoRR},
  volume    = {abs/1702.07392},
  year      = {2017},
  url       = {http://arxiv.org/abs/1702.07392},
  timestamp = {Wed, 01 Mar 2017 14:26:00 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/LiSEJ17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
