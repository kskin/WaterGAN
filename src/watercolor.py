from PIL import Image
import scipy
from scipy import misc
import numpy as np
import os
from scipy import io

# Script to convert in air to underwater images based on attenuation model only

# Read in air and depth
air_directory = "data/full_air_images/"
depth_directory =  "data/full_air_depth/"
water_directory ="results/"


# Convert to underwater
eta_r = -0.6212
eta_g = 0.2350
eta_b = -0.4676

for f_name in os.listdir(air_directory):
    if f_name.startswith("UW"):
        print(f_name)
        uw = np.zeros((640,480,3))
        din = scipy.io.loadmat(depth_directory+f_name[:8]+"mat")
        d =scipy.misc.imresize(din["depth"],(640,480),mode="F")
        d = np.expand_dims(d,axis=2)
        airin = scipy.misc.imread(air_directory+f_name)
        air = scipy.misc.imresize(airin,(640,480,3))

        uw[:,:,0] = np.multiply(air[:,:,0],np.exp(d[:,:,0]*eta_r))
        uw[:,:,1] = np.multiply(air[:,:,1],np.exp(d[:,:,0]*eta_g))
        uw[:,:,2] = np.multiply(air[:,:,2],np.exp(d[:,:,0]*eta_b))

        # Save underwater
        out_f_name = f_name
        scipy.misc.imsave(water_directory+out_f_name, uw)
