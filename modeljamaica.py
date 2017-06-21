from __future__ import division
import os
import PIL
import scipy.stats as st
import numpy as np
from PIL import Image
import time
import scipy
from scipy import misc
import scipy.misc
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import scipy.io as sio
from ops import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class WGAN(object):
  def __init__(self, sess, input_height=640, input_width=480, input_water_height=1360, input_water_width=1024, is_crop=True,
         batch_size=64, sample_num = 64, output_height=256, output_width=256,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,gfc_dim=1024, dfc_dim=1024, c_dim=3, max_depth=3.0,save_epoch = 100,
         water_dataset_name='default',air_dataset_name='default',
         depth_dataset_name='default',input_fname_pattern='*.png', checkpoint_dir=None, results_dir=None, sample_dir=None,num_samples=4000):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)
    self.batch_size = batch_size
    self.num_samples = num_samples

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width
    self.input_water_height = input_water_height
    self.input_water_width = input_water_width
    self.save_epoch = save_epoch
    self.y_dim = y_dim
    self.z_dim = z_dim
    self.max_depth=max_depth

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    self.sw = 640
    self.sh = 480

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')
      self.g_bn4 = batch_norm(name='g_bn4')

    self.water_dataset_name = water_dataset_name
    self.air_dataset_name = air_dataset_name
    self.depth_datset_name = depth_dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.results_dir = results_dir
    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    image_dims = [self.output_height, self.output_width, self.c_dim]
    sample_dims = [self.output_height,self.output_width, self.c_dim]
    self.water_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.air_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='air_images')
    self.depth_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.output_height,self.output_width,1], name='depth')
    self.water_sample_inputs = tf.placeholder(
     tf.float32, [self.num_samples] + image_dims, name='sample_inputs')
    self.depth_small_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.output_height,self.output_width,1], name='depth_small')

    self.R2 = tf.placeholder(tf.float32,[self.output_height,self.output_width], name='R2')
    self.R4 = tf.placeholder(tf.float32,[self.output_height,self.output_width], name='R4')
    self.R6 = tf.placeholder(tf.float32,[self.output_height,self.output_width], name='R6')

    self.sample_air_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.sh,self.sw,3], name='sample_air_images')
    self.sample_depth_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.sh,self.sw,1], name='sample_depth')
    self.sample_fake_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.sh,self.sw,3], name='sample_fake')

    sample_air_inputs = self.sample_air_inputs
    sample_depth_inputs = self.sample_depth_inputs
    depth_small_inputs = self.depth_small_inputs
    sample_fake_inputs = self.sample_fake_inputs

    water_inputs = self.water_inputs
    water_sample_inputs = self.water_sample_inputs
    air_inputs = self.air_inputs
    depth_inputs = self.depth_inputs

    R2 = self.R2
    R4 = self.R4
    R6 = self.R6

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = tf.summary.histogram("z", self.z)
    self.sample_z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.G,eta_r,eta_g,eta_b,C1,C2,C3,A = self.wc_generator(self.z,air_inputs, depth_inputs,R2,R4,R6)
    self.D, self.D_logits = self.discriminator(water_inputs)

    self.wc_sampler = self.wc_sampler(self.sample_z,sample_air_inputs, sample_depth_inputs,depth_small_inputs,R2,R4,R6)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    self.G_sum = tf.summary.image("G", self.G,max_outputs=200)

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits, targets=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, targets=tf.ones_like(self.D_)))
    self.c1_loss = -tf.minimum(tf.reduce_min(C1),0)*10000
    self.c2_loss = -tf.minimum(tf.reduce_min(-1*(4*C2*C2-12*C1*C3)),0)*10000

    self.eta_r_loss = -tf.minimum(tf.reduce_min(eta_r),0)*10000
    self.eta_g_loss = -tf.minimum(tf.reduce_min(eta_g),0)*10000
    self.eta_b_loss = -tf.minimum(tf.reduce_min(eta_b),0)*10000
    self.A_loss = -tf.minimum(tf.reduce_min(A),0)*10000
    self.g_loss = self.c1_loss + self.c2_loss + self.g_loss+ self.eta_r_loss + self.eta_g_loss +self.eta_b_loss + self.A_loss

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss) 
    self.D = tf.summary.scalar("D_realdata", self.D)
    self.D_ = tf.summary.scalar("D_fakedata", self.D_)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.saver = tf.train.Saver()

  def train(self, config):
    """Train WGAN"""
    water_data = glob(os.path.join("./data", config.water_dataset, self.input_fname_pattern))
    air_data = glob(os.path.join("./data", config.air_dataset, self.input_fname_pattern))
    depth_data = glob(os.path.join("./data", config.depth_dataset, "*.mat"))

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = tf.summary.merge([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    # Start training
    counter = 1
    start_time = time.time()
    errD_fake = 0.0
    errD_real = 0.0
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    k1 = np.ones([self.output_height,self.output_width],np.float32)
    r2 = np.ones([self.output_height,self.output_width],np.float32)
    r4 = np.ones([self.output_height,self.output_width],np.float32)
    r6 = np.ones([self.output_height,self.output_width],np.float32)

    #kernel = kernel.astype(np.float32)
    cx = self.output_width/2
    cy = self.output_height/2
    for i in range(0,self.output_height):
        for j in range(0,self.output_width):
            r = np.sqrt((i-cy)*(i-cy)+(j-cx)*(j-cx))/(np.sqrt(cy*cy+cx*cx))
            r2[i,j] = r*r
            r4[i,j] = r*r*r*r
            r6[i,j] = r*r*r*r*r*r
    #plt.imshow(r4, interpolation='none',cmap='Greys')
    #plt.savefig('test.png')
    print(r4.shape)
    print(r4.dtype)


    for epoch in xrange(config.epoch):
      checkprint=0
      water_data = sorted(glob(os.path.join(
        "./data", config.water_dataset, self.input_fname_pattern)))
      air_data = sorted(glob(os.path.join(
        "./data", config.air_dataset, self.input_fname_pattern)))
      depth_data = sorted(glob(os.path.join(
        "./data", config.depth_dataset, "*.mat")))

      water_batch_idxs = min(min(len(air_data),len(water_data)), config.train_size) // config.batch_size
      air_batch_idxs = water_batch_idxs
      randombatch = np.arange(water_batch_idxs*config.batch_size)
      np.random.shuffle(randombatch)
      # Load water images
      for idx in xrange(0, (water_batch_idxs*config.batch_size), config.batch_size):
        water_batch_files = []
        air_batch_files = []
        depth_batch_files = []

        for id in xrange(0, config.batch_size):
            water_batch_files = np.append(water_batch_files,water_data[randombatch[idx+id]])
            air_batch_files = np.append(air_batch_files,air_data[randombatch[idx+id]])
            depth_batch_files = np.append(depth_batch_files,depth_data[randombatch[idx+id]])
        #print(depth_batch_files)
        if self.is_crop:
            air_batch = [self.read_img(air_batch_file) for air_batch_file in air_batch_files]
            water_batch = [self.read_img(water_batch_file) for water_batch_file in water_batch_files]
            depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
        else:
            air_batch = [scipy.misc.imread(air_batch_file) for air_batch_file in air_batch_files]
            water_batch = [scipy.misc.imread(water_batch_file) for water_batch_file in water_batch_files]
            depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
        air_batch_images = np.array(air_batch).astype(np.float32)
        water_batch_images = np.array(water_batch).astype(np.float32)
        depth_batch_images = np.expand_dims(depth_batch,axis=3)
        r2 = np.array(r2).astype(np.float32)
        r4 = np.array(r4).astype(np.float32)
        r6 = np.array(r6).astype(np.float32)
        #print(r4.dtype)
        #print(r4.shape)

        batch_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.z: batch_z,self.water_inputs: water_batch_images,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images,self.R2:r2, self.R4:r4, self.R6:r6})
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z:batch_z, self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images,self.R2: r2, self.R4: r4, self.R6: r6})
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images, self.R2:r2, self.R4:r4, self.R6:r6})
        self.writer.add_summary(summary_str, counter)

        errD_fake = self.d_loss_fake.eval({self.z:batch_z, self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images,self.R2:r2, self.R4:r4, self.R6:r6})
        errD_real = self.d_loss_real.eval({self.water_inputs: water_batch_images})
        errG = self.g_loss.eval({self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images,self.R2:r2, self.R4:r4, self.R6:r6})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, water_batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        #if np.mod(counter, 5) == 1:
        if (1):
          print(self.sess.run('wc_generator/g_atten/g_eta_r:0'))
          print(self.sess.run('wc_generator/g_atten/g_eta_g:0'))
          print(self.sess.run('wc_generator/g_atten/g_eta_b:0'))
          print(self.sess.run('wc_generator/g_vig/g_amp:0'))
          print(self.sess.run('wc_generator/g_vig/g_c1:0'))
          print(self.sess.run('wc_generator/g_vig/g_c2:0'))
          print(self.sess.run('wc_generator/g_vig/g_c3:0'))


        if (epoch == self.save_epoch) and (checkprint == 0):
        # Load samples in batches of 100
            checkprint = 1
            self.save(config.checkpoint_dir, counter)
            print("saving checkpoint")

            sample_batch_idxs = self.num_samples // config.batch_size
            #print(sample_batch_idxs)
            for idx in xrange(0, sample_batch_idxs):
                sample_water_batch_files = water_data[idx*config.batch_size:(idx+1)*config.batch_size]
                sample_air_batch_files = air_data[idx*config.batch_size:(idx+1)*config.batch_size]
                sample_depth_batch_files = depth_data[idx*config.batch_size:(idx+1)*config.batch_size]
                if self.is_crop:
                    sample_air_batch = [self.read_img_sample(sample_air_batch_file) for sample_air_batch_file in sample_air_batch_files]
                    sample_water_batch = [self.read_img_sample(sample_water_batch_file) for sample_water_batch_file in sample_water_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                else:
                    sample_air_batch = [scipy.misc.imread(sample_air_batch_file) for sample_air_batch_file in sample_air_batch_files]
                    sample_water_batch = [scipy.misc.imread(sample_water_batch_file) for sample_water_batch_file in sample_water_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                sample_air_images = np.array(sample_air_batch).astype(np.float32)
                sample_water_images = np.array(sample_water_batch).astype(np.float32)
                sample_depth_small_images = np.expand_dims(sample_depth_small_batch,axis=3)
                sample_depth_images = np.expand_dims(sample_depth_batch,axis=3)
                sample_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)

                samples = self.sess.run([self.wc_sampler],
                    feed_dict={self.sample_z:sample_z,self.sample_air_inputs:sample_air_images,
                    self.sample_depth_inputs: sample_depth_images,self.depth_small_inputs:sample_depth_small_images,self.R2:r2, self.R4:r4, self.R6:r6})
                sample_ims = np.asarray(samples)
                sample_ims = np.squeeze(sample_ims)
                sample_fake_images = sample_ims[:,0:self.sh,0:self.sw,0:3]
                sample_fake_images_small = np.empty([0,self.sh,self.sw,3])
                for img_idx in range(0,self.batch_size):
                    out_file = "/fake_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                    out_name = self.results_dir + out_file
                    print(out_name)
                    sample_im = sample_ims[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_im = np.squeeze(sample_im)
                    try:
                      scipy.misc.imsave(out_name,sample_im)
                    except OSError:
                      print(out_name)
                      print("ERROR!")
                      pass
                    out_file2 = "/air_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                    out_name2 = self.results_dir + out_file2
                    sample_im2 = sample_air_images[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_im2 = np.squeeze(sample_im2)
                    try:
                      scipy.misc.imsave(out_name2,sample_im2)
                    except OSError:
                      print(out_name)
                      print("ERROR!")
                      pass
                    out_file3 = "/depth_%0d_%02d_%02d.mat" % (epoch, img_idx,idx)
                    out_name3 = self.results_dir + out_file3
                    sample_im3 = sample_depth_images[img_idx,0:self.sh,0:self.sw,0]
                    sample_im3 = np.squeeze(sample_im3)
                    try:
                      sio.savemat(out_name3,{'depth':sample_im3})
                    except OSError:
                      print(out_name)
                      print("ERROR!")
                      pass
                    sample_fake = sample_fake_images[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_fake = np.squeeze(sample_fake)
                    sample_fake = scipy.misc.imresize(sample_fake,[self.sh,self.sw,3],interp='bicubic')
                    sample_fake = np.expand_dims(sample_fake,axis=0)
                    sample_fake_images_small = np.append(sample_fake_images_small, sample_fake, axis=0)
        if (np.mod(epoch, 2) == 0) and (idx == 0):
          self.save(config.checkpoint_dir, counter)
          print("saving checkpoint")

  def test(self, config):
    """Train WGAN"""
    water_data = glob(os.path.join("./data", config.water_dataset, self.input_fname_pattern))
    air_data = glob(os.path.join("./data", config.air_dataset, self.input_fname_pattern))
    depth_data = glob(os.path.join("./data", config.depth_dataset, "*.mat"))

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = tf.summary.merge([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    # Start training
    counter = 1
    start_time = time.time()
    errD_fake = 0.0
    errD_real = 0.0
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    k1 = np.ones([self.output_height,self.output_width],np.float32)
    r2 = np.ones([self.output_height,self.output_width],np.float32)
    r4 = np.ones([self.output_height,self.output_width],np.float32)
    r6 = np.ones([self.output_height,self.output_width],np.float32)

    cx = self.output_width/2
    cy = self.output_height/2
    for i in range(0,self.output_height):
        for j in range(0,self.output_width):
            r = np.sqrt((i-cy)*(i-cy)+(j-cx)*(j-cx))/(np.sqrt(cy*cy+cx*cx))
            r2[i,j] = r*r
            r4[i,j] = r*r*r*r
            r6[i,j] = r*r*r*r*r*r


    for epoch in xrange(config.epoch):
      checkprint=0
      water_data = sorted(glob(os.path.join(
        "./data", config.water_dataset, self.input_fname_pattern)))
      air_data = sorted(glob(os.path.join(
        "./data", config.air_dataset, self.input_fname_pattern)))
      depth_data = sorted(glob(os.path.join(
        "./data", config.depth_dataset, "*.mat")))

      water_batch_idxs = min(min(len(air_data),len(water_data)), config.train_size) // config.batch_size
      air_batch_idxs = water_batch_idxs
      randombatch = np.arange(water_batch_idxs*config.batch_size)
      np.random.shuffle(randombatch)

      if(1):
        if (1):
          print(self.sess.run('wc_generator/g_atten/g_eta_r:0'))
          print(self.sess.run('wc_generator/g_atten/g_eta_g:0'))
          print(self.sess.run('wc_generator/g_atten/g_eta_b:0'))
          print(self.sess.run('wc_generator/g_vig/g_amp:0'))
          print(self.sess.run('wc_generator/g_vig/g_c1:0'))
          print(self.sess.run('wc_generator/g_vig/g_c2:0'))
          print(self.sess.run('wc_generator/g_vig/g_c3:0'))

        # Load samples in batches of 100
        if (1):
            checkprint = 1

            sample_batch_idxs = self.num_samples // config.batch_size
            for idx in xrange(0, sample_batch_idxs):
                sample_water_batch_files = water_data[idx*config.batch_size:(idx+1)*config.batch_size]
                sample_air_batch_files = air_data[idx*config.batch_size:(idx+1)*config.batch_size]
                sample_depth_batch_files = depth_data[idx*config.batch_size:(idx+1)*config.batch_size]
                if self.is_crop:
                    sample_air_batch = [self.read_img_sample(sample_air_batch_file) for sample_air_batch_file in sample_air_batch_files]
                    sample_water_batch = [self.read_img_sample(sample_water_batch_file) for sample_water_batch_file in sample_water_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                else:
                    sample_air_batch = [scipy.misc.imread(sample_air_batch_file) for sample_air_batch_file in sample_air_batch_files]
                    sample_water_batch = [scipy.misc.imread(sample_water_batch_file) for sample_water_batch_file in sample_water_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                sample_air_images = np.array(sample_air_batch).astype(np.float32)
                sample_water_images = np.array(sample_water_batch).astype(np.float32)
                sample_depth_small_images = np.expand_dims(sample_depth_small_batch,axis=3)
                sample_depth_images = np.expand_dims(sample_depth_batch,axis=3)
                sample_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)
                samples = self.sess.run([self.wc_sampler],
                    feed_dict={self.sample_z:sample_z,self.sample_air_inputs:sample_air_images,
                    self.sample_depth_inputs: sample_depth_images,self.depth_small_inputs:sample_depth_small_images,self.R2:r2, self.R4:r4, self.R6:r6})
                sample_ims = np.asarray(samples)
                sample_ims = np.squeeze(sample_ims)
                sample_fake_images = sample_ims[:,0:self.sh,0:self.sw,0:3]
                sample_fake_images_small = np.empty([0,self.sh,self.sw,3])
                for img_idx in range(0,self.batch_size):
                    out_file = "/fake_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                    out_name = self.results_dir + out_file
                    print(out_name)
                    sample_im = sample_ims[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_im = np.squeeze(sample_im)
                    try:
                      scipy.misc.imsave(out_name,sample_im)
                    except OSError:
                      print(out_name)
                      print("ERROR!")
                      pass
                    out_file2 = "/air_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                    out_name2 = self.results_dir + out_file2
                    sample_im2 = sample_air_images[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_im2 = np.squeeze(sample_im2)
                    try:
                      scipy.misc.imsave(out_name2,sample_im2)
                    except OSError:
                      print(out_name)
                      print("ERROR!")
                      pass
                    out_file3 = "/depth_%0d_%02d_%02d.mat" % (epoch, img_idx,idx)
                    out_name3 = self.resuts_dir + out_file3
                    sample_im3 = sample_depth_images[img_idx,0:self.sh,0:self.sw,0]
                    sample_im3 = np.squeeze(sample_im3)
                    try:
                      sio.savemat(out_name3,{'depth':sample_im3})
                    except OSError:
                      print(out_name)
                      print("ERROR!")
                      pass
                    sample_fake = sample_fake_images[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_fake = np.squeeze(sample_fake)
                    sample_fake = scipy.misc.imresize(sample_fake,[self.sh,self.sw,3],interp='bicubic')
                    sample_fake = np.expand_dims(sample_fake,axis=0)
                    sample_fake_images_small = np.append(sample_fake_images_small, sample_fake, axis=0)
        if (np.mod(epoch, 5) == 1) and (idx == 0):
          self.save(config.checkpoint_dir, counter)
          print("saving checkpoint")

  def discriminator(self, image, depth=None,y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4), h4


  def sample_discriminator(self, image, depth=None,y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4)


  def wc_generator(self, z, image, depth,r2,r4,r6, y=None):
    with tf.variable_scope("wc_generator") as scope:

      # water-based attenuation and backscatter
      with tf.variable_scope("g_atten"):
          init_r = tf.random_normal([1,1,1],mean=0.35,stddev=0.01,dtype=tf.float32)
          eta_r = tf.get_variable("g_eta_r",initializer=init_r)
          init_b = tf.random_normal([1,1,1],mean=0.0194,stddev=0.01,dtype=tf.float32)
          eta_b = tf.get_variable("g_eta_b",initializer=init_b)
          init_g = tf.random_normal([1,1,1],mean=0.038,stddev=0.01,dtype=tf.float32)
          eta_g = tf.get_variable("g_eta_g",initializer=init_g)
          eta = tf.pack([eta_r,eta_g,eta_b],axis=3)
          eta_d = tf.exp(tf.mul(-1.0,tf.mul(depth,eta)))

      h0 = tf.mul(image,eta_d)

     # backscattering
      self.z_, self.h0z_w, self.h0z_b = linear(
          z, self.output_width*self.output_height*self.batch_size*1, 'g_h0_lin', with_w=True)

      self.h0z = tf.reshape(
          self.z_, [-1, self.output_height, self.output_width, self.batch_size*1])
      h0z = tf.nn.relu(self.g_bn0(self.h0z))
      h0z = tf.multiply(h0z,depth)

      with tf.variable_scope('g_h1_conv'):
          w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1z = tf.nn.conv2d(h0z, w, strides=[1, 1,1, 1], padding='SAME')
      h_g = lrelu(self.g_bn1(h1z))

      with tf.variable_scope('g_h1_convr'):
          wr = tf.get_variable('g_wr',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 1, 1, 1], padding='SAME')
      h_r = lrelu(self.g_bn3(h1zr))

      with tf.variable_scope('g_h1_convb'):
          wb = tf.get_variable('g_wb',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zb = tf.nn.conv2d(h0z, wb, strides=[1, 1, 1, 1], padding='SAME')
      h_b = lrelu(self.g_bn4(h1zb))

      h_r = tf.squeeze(h_r,axis=3)
      h_g = tf.squeeze(h_g,axis=3)
      h_b = tf.squeeze(h_b,axis=3)

      h_final=tf.pack([h_r,h_g,h_b],axis=3)

      h2 = tf.add(h_final,h0)

      # camera model
      with tf.variable_scope("g_vig"):
          A = tf.get_variable('g_amp', [1],
              initializer=tf.truncated_normal_initializer(mean=0.9,stddev=0.01))
          C1 = tf.get_variable('g_c1', [1],
              initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
          C2 = tf.get_variable('g_c2', [1],
              initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
          C3 = tf.get_variable('g_c3', [1],
              initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
      h11 = tf.multiply(r2,C1)
      h22 = tf.multiply(r4,C2)
      h33 = tf.multiply(r6,C3)
      h44 = tf.ones([self.output_height,self.output_width],tf.float32)
      h1 = tf.add(tf.add(h44,h11),tf.add(h22,h33))
      V = tf.expand_dims(h1,axis=2)
      h1a = tf.divide(h2,V)
      h_out = tf.multiply(h1a,A)
      return h_out, eta_r,eta_g,eta_b, C1,C2,C3,A

  def wc_sampler(self, z, image, depth, depth_small,r2,r4,r6,y=None):
    with tf.variable_scope("wc_generator",reuse=True) as scope:
      # water-based attenuation
      with tf.variable_scope("g_atten",reuse=True):
          init_r = tf.random_normal([1,1,1],mean=0.35,stddev=0.01,dtype=tf.float32)
          eta_r = tf.get_variable("g_eta_r",initializer=init_r)
          init_b = tf.random_normal([1,1,1],mean=0.0194,stddev=0.01,dtype=tf.float32)
          eta_b = tf.get_variable("g_eta_b",initializer=init_b)
          init_g = tf.random_normal([1,1,1],mean=0.038,stddev=0.01,dtype=tf.float32)
          eta_g = tf.get_variable("g_eta_g",initializer=init_g)
          eta = tf.pack([eta_r,eta_g,eta_b],axis=3)

          eta_d = tf.exp(tf.mul(-1.0,tf.mul(depth,eta)))
          h0 = tf.mul(image,eta_d)

      self.z_, self.h0z_w, self.h0z_b = linear(
          z, self.output_width*self.output_height*self.batch_size*1, 'g_h0_lin', with_w=True)

      self.h0z = tf.reshape(
          self.z_, [-1, self.output_height, self.output_width, self.batch_size*1])
      h0z = tf.nn.relu(self.g_bn0(self.h0z))
      h0z = tf.multiply(h0z,depth_small)

      # backscattering
      with tf.variable_scope('g_h1_conv',reuse=True):
          w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1z = tf.nn.conv2d(h0z, w, strides=[1, 1, 1, 1], padding='SAME')
      h_g = lrelu(self.g_bn1(h1z))

      with tf.variable_scope('g_h1_convr',reuse=True):
          wr = tf.get_variable('g_wr',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 1, 1, 1], padding='SAME')
      h_r = lrelu(self.g_bn3(h1zr))

      with tf.variable_scope('g_h1_convb',reuse=True):
          wb = tf.get_variable('g_wb',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zb = tf.nn.conv2d(h0z, wb, strides=[1,1,1, 1], padding='SAME')
      h_b = lrelu(self.g_bn4(h1zb))

      h_r1 = tf.image.resize_images(h_r,[120,160],method=2)
      h_g1 = tf.image.resize_images(h_g,[120,160],method=2)
      h_b1 = tf.image.resize_images(h_b,[120,160],method=2)
      h_rxlt = tf.image.resize_images(h_r1,[240,320],method=2)
      h_gxlt = tf.image.resize_images(h_g1,[240,320],method=2)
      h_bxlt = tf.image.resize_images(h_b1,[240,320],method=2)

      h_rxl = tf.image.resize_images(h_rxlt,[480,640],method=2)
      h_gxl = tf.image.resize_images(h_gxlt,[480,640],method=2)
      h_bxl = tf.image.resize_images(h_bxlt,[480,640],method=2)

      h_rxl = tf.squeeze(h_rxl,axis=3)
      h_gxl = tf.squeeze(h_gxl,axis=3)
      h_bxl = tf.squeeze(h_bxl,axis=3)
      h_final=tf.pack([h_rxl,h_gxl,h_bxl],axis=3)
      h2 = tf.add(h_final,h0)

     # camera model
      with tf.variable_scope("g_vig",reuse=True):
          A = tf.get_variable('g_amp', [1],
              initializer=tf.truncated_normal_initializer(mean=0.9,stddev=0.01))
          C1 = tf.get_variable('g_c1', [1],
              initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
          C2 = tf.get_variable('g_c2', [1],
              initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
          C3 = tf.get_variable('g_c3', [1],
              initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))

      h11 = tf.multiply(r2,C1)
      h22 = tf.multiply(r4,C2)
      h33 = tf.multiply(r6,C3)
      h44 = tf.ones([self.output_height,self.output_width],tf.float32)
      h1 = tf.add(tf.add(h44,h11),tf.add(h22,h33))
      V = tf.expand_dims(h1,axis=2)
      h1a = V
      h1a1 = tf.image.resize_images(h1a,[120,160],method=2)
      h1_xlt = tf.image.resize_images(h1a1,[240,320],method=2)
      h1_xl = tf.image.resize_images(h1_xlt,[480,640],method=2)
      h_out1 = tf.divide(h2,h1_xl)
      h_out = tf.multiply(h_out1,A)
      return h_out

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.water_dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False

  def read_depth(self, filename):
    depth_mat = sio.loadmat(filename)
    depthtmp=depth_mat["depth"]
    ds = depthtmp.shape
    if self.is_crop:
      depth = scipy.misc.imresize(depthtmp,(self.output_height,self.output_width),mode='F')
    depth = np.array(depth).astype(np.float32)
    depth = np.multiply(self.max_depth,np.divide(depth,depth.max()))

    return depth

  def read_img(self, filename):
    imgtmp = scipy.misc.imread(filename)
    ds = imgtmp.shape
    if self.is_crop:
      img = scipy.misc.imresize(imgtmp,(self.output_height,self.output_width,3))
    img = np.array(img).astype(np.float32)
    return img


  def read_depth_small(self, filename):
    depth_mat = sio.loadmat(filename)
    depthtmp=depth_mat["depth"]
    ds = depthtmp.shape

    if self.is_crop:
      depth = scipy.misc.imresize(depthtmp,(self.output_height,self.output_width),mode='F')
    depth = np.array(depth).astype(np.float32)
    depth = np.multiply(self.max_depth,np.divide(depth,depth.max()))

    return depth

  def read_depth_sample(self, filename):
    depth_mat = sio.loadmat(filename)
    depthtmp=depth_mat["depth"]
    ds = depthtmp.shape
    if self.is_crop:
      depth = scipy.misc.imresize(depthtmp,(self.sh,self.sw),mode='F')
    depth = np.array(depth).astype(np.float32)
    depth = np.multiply(self.max_depth,np.divide(depth,depth.max()))

    return depth

  def read_img_sample(self, filename):
    imgtmp = scipy.misc.imread(filename)
    ds = imgtmp.shape
    if self.is_crop:
      img = scipy.misc.imresize(imgtmp,(self.sh,self.sw,3))
    img = np.array(img).astype(np.float32)
    return img


