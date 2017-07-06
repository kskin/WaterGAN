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

If you find this work useful for your research, please cite WaterGAN in your publications.

```
@article{Li:2017aa,
	Author = {Jie Li and Katherine A. Skinner and Ryan Eustice and M. Johnson-Roberson},
	Date-Added = {2017-06-12 22:07:13 +0000},
	Date-Modified = {2017-06-12 22:12:20 +0000},
	Journal = {IEEE Robotics and Automation Letters (RA-L)},
	Keywords = {jrnl},
	Note = {accepted},
	Title = {WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images},
	Year = {2017}}
```
