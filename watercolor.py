from PIL import Image
import scipy
from scipy import misc
import numpy as np
import os
from scipy import io

# Read in air and depth
air_directory = "data/full_air_images/"
depth_directory =  "data/full_air_depth/"
water_directory ="results/"


# Convert to underwater
eta_r = -0.6212
eta_g = 0.2350
eta_b = -0.4676
#beta = -0.0389

for f_name in os.listdir(air_directory):
    if f_name.startswith("UW"):
        print(f_name)
        uw = np.zeros((640,480,3))
        din = scipy.io.loadmat(depth_directory+f_name[:8]+"mat")
        d =scipy.misc.imresize(din["depth"],(640,480),mode="F")
        d = np.expand_dims(d,axis=2)
        airin = scipy.misc.imread(air_directory+f_name)
        air = scipy.misc.imresize(airin,(640,480,3))
        #print(air.shape)
        #print(d.shape)

        uw[:,:,0] = np.multiply(air[:,:,0],np.exp(d[:,:,0]*eta_r))
        uw[:,:,1] = np.multiply(air[:,:,1],np.exp(d[:,:,0]*eta_g))
        uw[:,:,2] = np.multiply(air[:,:,2],np.exp(d[:,:,0]*eta_b))

  #      for i in range(0,640):
   #         for j in range(0,480):
    #            uw[i,j,0] = air[i,j,0]*np.exp(d[i,j]*eta_r)
     #           uw[i,j,1] = air[i,j,1]*np.exp(d[i,j]*eta_g)
      #          uw[i,j,2] = air[i,j,2]*np.exp(d[i,j]*eta_b)

        # Save underwater
        out_f_name = f_name
        scipy.misc.imsave(water_directory+out_f_name, uw)
