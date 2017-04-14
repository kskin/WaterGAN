# WaterGAN Tutorial

Work in progres...

<p align="center">
  <img src="https://github.com/kskin/WaterGAN/blob/master/watergan.PNG?raw=true"/>
</p>

![](watergan.PNG)

+ This repository contains source code for WaterGAN developed in [WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images](https://arxiv.org/abs/1702.07392).
+ This code is modified from [Taehoon Kim's](http://carpedm20.github.io/)
  [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) (MIT-licensed). Our modifications are [MIT-licensed](./LICENSE).

# Usage

Download data:

Coming soon...

Train a model:

```
$ python main.py --water_dataset water_images --air_dataset air_images --depth_dataset air_depth
```

# Results

Original in-air images:

![](figures/air-raw.png)

Synthetic underwater images produced by WaterGAN:

![](figures/air-gen.png)

Now we have a dataset with paired true color, depth, and (synthetic) underwater images. We use this to train an end-to-end network for underwater image restoration. For more details, see the [paper].(https://arxiv.org/abs/1702.07392) 

Raw underwater image gathered from a survey in a pure water tank:

![](figures/mhl-raw.png)

Corrected images using data generated with WaterGAN to train an end-to-end image restoration pipeline:

![](figures/mhl-corrected.png)
  
# Citations

```
@article{li2017watergan,
    author    = {Jie Li and
               Katherine A. Skinner and
               Ryan M. Eustice and
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
