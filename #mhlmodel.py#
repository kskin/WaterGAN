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

class WGAN(object):
  def __init__(self, sess, input_height=64, input_width=64, input_water_height=64, input_water_width=64, is_crop=False,
         is_stereo=False,batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,gfc_dim=1024, dfc_dim=1024, c_dim=3, max_depth=3.0,save_epoch = 100,
         water_dataset_name='default',air_dataset_name='default',
         depth_dataset_name='default',waterdepth_dataset_name='default',input_fname_pattern='*.png', checkpoint_dir=None, sample_dir=None,num_samples=4000):
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
    self.is_stereo = is_stereo
    self.batch_size = batch_size
    #self.sample_num = sample_num
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

    self.sw = 256
    self.sh = 256

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
    self.waterdepth_dataset_name = waterdepth_dataset_name
    self.air_dataset_name = air_dataset_name
    self.depth_datset_name = depth_dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    image_dims = [self.output_height, self.output_width, self.c_dim]
    sample_dims = [64,64, self.c_dim]
    self.water_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.air_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='air_images')
    self.depth_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.output_height,self.output_width,1], name='depth')
    self.water_sample_inputs = tf.placeholder(
      tf.float32, [self.num_samples] + image_dims, name='sample_inputs')
    self.waterdepth_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.output_height,self.output_width,1], name='waterdepth')
    self.depth_small_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [64,64,1], name='depth_small')


    self.sample_air_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.sw,self.sh,3], name='air_images')
    self.sample_depth_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.sw,self.sh,1], name='depth')
    self.sample_fake_inputs = tf.placeholder(
      tf.float32, [self.batch_size] + [self.sw,self.sh,3], name='depth')

    sample_air_inputs = self.sample_air_inputs
    sample_depth_inputs = self.sample_depth_inputs
    depth_small_inputs = self.depth_small_inputs
    sample_fake_inputs = self.sample_fake_inputs

    water_inputs = self.water_inputs
    water_sample_inputs = self.water_sample_inputs
    air_inputs = self.air_inputs
    depth_inputs = self.depth_inputs
    waterdepth_inputs = self.waterdepth_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = tf.summary.histogram("z", self.z)
    self.sample_z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.G,eta_r,eta_g,eta_b,beta = self.wc_generator(self.z,air_inputs, depth_inputs)
    if self.is_stereo:
        self.D, self.D_logits = self.discriminator(water_inputs,depth_inputs)
    else:
        self.D, self.D_logits = self.discriminator(water_inputs)

    #self.sample_discriminator = self.sample_discriminator(sample_fake_inputs,reuse=True)

    print(self.D_logits)
    self.wc_sampler = self.wc_sampler(self.sample_z,sample_air_inputs, sample_depth_inputs,depth_small_inputs)
    #self.wc_sampler = self.wc_sampler(self.z,air_inputs, depth_inputs)
    if self.is_stereo:
        self.D_, self.D_logits_ = self.discriminator(self.G,depth_inputs, reuse=True)
    else:
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    print(self.D_)
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
    # constrain eta > 0
    self.eta_r_loss = -tf.minimum(tf.reduce_min(eta_r),0)*1000
    self.eta_g_loss = -tf.minimum(tf.reduce_min(eta_g),0)*1000
    self.eta_b_loss = -tf.minimum(tf.reduce_min(eta_b),0)*1000
    self.eta_r_loss1 = -tf.minimum(tf.reduce_min((1.0-eta_r)),0)*1000
    self.eta_g_loss1 = -tf.minimum(tf.reduce_min((1.0-eta_g)),0)*1000
    self.eta_b_loss1 = -tf.minimum(tf.reduce_min((1.0-eta_b)),0)*1000

    self.beta_loss = -tf.minimum(tf.reduce_min(beta),0)
    self.g_loss = self.g_loss+ self.eta_r_loss + self.eta_g_loss +self.eta_b_loss + self.eta_r_loss1 + self.eta_g_loss1 +self.eta_b_loss1

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
    waterdepth_data = glob(os.path.join("./data", config.waterdepth_dataset, "*.mat"))
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


    #water_data = sorted(glob(os.path.join(
    #  "./data", config.water_dataset, self.input_fname_pattern)))
    #air_data = sorted(glob(os.path.join(
    #  "./data", config.air_dataset, self.input_fname_pattern)))
    #depth_data = sorted(glob(os.path.join(
    #  "./data", config.depth_dataset, "*.mat")))


    #print("training...")
    #num_samples = min(len(air_data),len(water_data))

    #sample_water_files = water_data[0:num_samples]
    #sample_air_files = air_data[0:num_samples]
    #sample_depth_files = depth_data[0:num_samples]
    #if self.is_crop:
    #    sample_air = [scipy.misc.imresize(scipy.misc.imread(air_file),(self.output_height,self.output_width,3)) for air_file in sample_air_files]
    #    sample_water = [scipy.misc.imresize(scipy.misc.imread(water_file),(self.output_height,self.output_width,3)) for water_file in sample_water_files]
    #    sample_depth = [scipy.misc.imresize(sio.loadmat(depth_file)["depth"],(self.output_height,self.output.width),mode='F') for depth_file in sample_depth_files]

    #else:
    ##    sample_air = [scipy.misc.imread(air_file) for air_file in air_data]
    #    sample_water = [scipy.misc.imread(water_file) for water_file in water_data]
    #    sample_depth = [sio.loadmat(depth_file)["depth"] for depth_file in depth_data]
    #sample_air_images = np.array(sample_air).astype(np.float32)
    #sample_water_images = np.array(sample_water).astype(np.float32)
    #sample_depth_images = np.expand_dims(sample_depth,axis=3)
    #sample_z = np.random.uniform(-1,1,[num_samples,self.z_dim]).astype(np.float32)
    #print("loading samples...")
    #sample_z = np.random.uniform(-1,1,size=(self.sample_num,self.z_dim))

    # Load data
    #sample_files = water_data[0:self.sample_num]
    #sample = [
    #    get_image(sample_file,
    #              input_height=self.input_height,
    #              input_width=self.input_width,
    #              resize_height=self.output_height,
    ##              resize_width=self.output_width,
    #              is_crop=self.is_crop,
    #              is_grayscale=self.is_grayscale) for sample_file in sample_files]
    #water_sample_inputs = np.array(sample).astype(np.float32)

    # Start training
    counter = 1
    start_time = time.time()
    errD_fake = 0.0
    errD_real = 0.0
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    #checkprint=0
    for epoch in xrange(config.epoch):
      checkprint=0
      water_data = sorted(glob(os.path.join(
        "./data", config.water_dataset, self.input_fname_pattern)))
      air_data = sorted(glob(os.path.join(
        "./data", config.air_dataset, self.input_fname_pattern)))
      depth_data = sorted(glob(os.path.join(
        "./data", config.depth_dataset, "*.mat")))
      waterdepth_data = sorted(glob(os.path.join(
        "./data", config.waterdepth_dataset, "*.mat")))

      water_batch_idxs = min(min(len(air_data),len(water_data)), config.train_size) // config.batch_size
      air_batch_idxs = water_batch_idxs
      randombatch = np.arange(water_batch_idxs*config.batch_size)
      np.random.shuffle(randombatch)
      # Load water images
      for idx in xrange(0, (water_batch_idxs*config.batch_size), config.batch_size):
      #for id in range(0,len(randombatch)):
        water_batch_files = []
        air_batch_files = []
        depth_batch_files = []
        waterdepth_batch_files = []


        for id in xrange(0, config.batch_size):
            water_batch_files = np.append(water_batch_files,water_data[randombatch[idx+id]])
            #print(water_batch_files)
            #print(water_data[585])
            air_batch_files = np.append(air_batch_files,air_data[randombatch[idx+id]])
            depth_batch_files = np.append(depth_batch_files,depth_data[randombatch[idx+id]])
            waterdepth_batch_files = np.append(waterdepth_batch_files,waterdepth_data[randombatch[idx+id]])

       # water_batch_files = water_data[idx*config.batch_size:(idx+1)*config.batch_size]
       # air_batch_files = air_data[idx*config.batch_size:(idx+1)*config.batch_size]
       # depth_batch_files = depth_data[idx*config.batch_size:(idx+1)*config.batch_size]
       # waterdepth_batch_files = waterdepth_data[idx*config.batch_size:(idx+1)*config.batch_size]
        if self.is_crop:
            air_batch = [scipy.misc.imresize(scipy.misc.imread(air_batch_file),(self.output_height,self.output_width,3)) for air_batch_file in air_batch_files]
            water_batch = [scipy.misc.imresize(scipy.misc.imread(water_batch_file),(self.output_height,self.output_width,3)) for water_batch_file in water_batch_files]
            #depth_batch = [scipy.misc.imresize(sio.loadmat(depth_batch_file)["depth"],(self.output_height,self.output_width),mode='F') for depth_batch_file in depth_batch_files]
            depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
            waterdepth_batch = [scipy.misc.imresize(sio.loadmat(waterdepth_batch_file)["depth"],(self.output_height,self.output_width),mode='F') for waterdepth_batch_file in waterdepth_batch_files]
        else:
            air_batch = [scipy.misc.imread(air_batch_file) for air_batch_file in air_batch_files]
            water_batch = [scipy.misc.imread(water_batch_file) for water_batch_file in water_batch_files]
            #depth_batch = [sio.loadmat(depth_batch_file)["depth"] for depth_batch_file in depth_batch_files]
            depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
            waterdepth_batch = [sio.loadmat(waterdepth_batch_file)["depth"] for waterdepth_batch_file in waterdepth_batch_files]
        air_batch_images = np.array(air_batch).astype(np.float32)
        water_batch_images = np.array(water_batch).astype(np.float32)
        #print ("Normalizing depth...")
        #print(depth_batch.max())
        #print(depth_batch.max()/self.max_depth)
        #depth_batch *= (depth_batch.max()/self.max_depth)
        #depth_batch_f =np.array(depth_batch).astype(np.float32)
        #depth_batch_n = np.multiply(self.max_depth,np.divide(depth_batch_f,depth_batch_f.max()))
        #print (depth_batch_n)
        depth_batch_images = np.expand_dims(depth_batch,axis=3)
        waterdepth_batch_images = np.expand_dims(waterdepth_batch,axis=3)

        batch_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)

        # Update D network
      #  with tf.device('/gpu:2'):
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.z: batch_z,self.water_inputs: water_batch_images,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images, self.waterdepth_inputs:waterdepth_batch_images})
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z:batch_z, self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})
        self.writer.add_summary(summary_str, counter)

        #_, summary_str = self.sess.run([g_optim, self.g_sum],
        #  feed_dict={self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})
        #self.writer.add_summary(summary_str, counter)


        errD_fake = self.d_loss_fake.eval({self.z:batch_z, self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})
        errD_real = self.d_loss_real.eval({self.water_inputs: water_batch_images,self.waterdepth_inputs:waterdepth_batch_images })
        errG = self.g_loss.eval({self.z:batch_z,self.air_inputs: air_batch_images,self.depth_inputs:depth_batch_images})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, water_batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 5) == 1:
          print(self.sess.run('wc_generator/g_atten/g_eta_r:0'))
          print(self.sess.run('wc_generator/g_atten/g_eta_g:0'))
          print(self.sess.run('wc_generator/g_atten/g_eta_b:0'))
         # print(self.sess.run('discriminator/Sigmoid:0'))
         # print(self.sess.run('discriminator_1/Sigmoid:0'))
          #print(self.sess.run('wc_generator/g_atten/g_beta_g:0'))
          print(self.sess.run('wc_generator/g_vig/g_amp:0'))
          #print(self.sess.run('wc_generator/g_atten/g_beta:0'))
          #print(self.sess.run('wc_generator/g_vig/g_w:0'))
          #print(self.sess.run('wc_generator/g_h1_deconv


#print(self.sess.run('wc_generator/g_vig/w:0'))

          #if config.water_dataset == 'mnist':
          #  print("oops")
          #else:
            #try:
         # d_loss, g_loss = self.sess.run([self.d_loss, self.g_loss],
         #     feed_dict={self.z: sample_z,
          #        self.air_inputs: sample_air_images,
           #       self.depth_inputs: sample_depth_images ,
            #      self.water_inputs: sample_water_images })
#        if (np.mod(epoch,5)==1) and (checkprint==0):
        #print(self.D)
        #print(self.D_)
        if (epoch == self.save_epoch) and (checkprint == 0):
#            # Load samples in batches of 100
            #print(self.num_samples)

            checkprint = 1
            sample_batch_idxs = self.num_samples // config.batch_size
            #print(sample_batch_idxs)
            for idx in xrange(0, sample_batch_idxs):
                sample_water_batch_files = water_data[idx*config.batch_size:(idx+1)*config.batch_size]
                sample_air_batch_files = air_data[idx*config.batch_size:(idx+1)*config.batch_size]
                sample_depth_batch_files = depth_data[idx*config.batch_size:(idx+1)*config.batch_size]
                #sample_waterdepth_batch_files = waterdepth_data[idx*100:(idx+1)*100]
                if self.is_crop:
                    sample_air_batch = [scipy.misc.imresize(scipy.misc.imread(sample_air_batch_file),(self.sw,self.sh,3)) for sample_air_batch_file in sample_air_batch_files]
                    sample_water_batch = [scipy.misc.imresize(scipy.misc.imread(sample_water_batch_file),(self.sw,self.sh,3)) for sample_water_batch_file in sample_water_batch_files]
                    #sample_depth_batch = [scipy.misc.imresize(sio.loadmat(sample_depth_batch_file)["depth"],(self.output_height,self.output_width),mode='F') for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    #sample_waterdepth_batch = [scipy.misc.imresize(sio.loadmat(sample_waterdepth_batch_file)["depth"],(self.output_height,self.output_width),mode='F') for sample_waterdepth_batch_file in sample_waterdepth_batch_files]
                else:
                    sample_air_batch = [scipy.misc.imread(sample_air_batch_file) for sample_air_batch_file in sample_air_batch_files]
                    sample_water_batch = [scipy.misc.imread(sample_water_batch_file) for sample_water_batch_file in sample_water_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    #sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for sample_depth_batch_file in sample_depth_batch_files]
                    #sample_depth_batch = [sio.loadmat(sample_depth_batch_file)["depth"] for sample_depth_batch_file in sample_depth_batch_files]
                    #sample_waterdepth_batch = [sio.loadmat(sample_waterdepth_batch_file)["depth"] for sample_waterdepth_batch_file in sample_waterdepth_batch_files]
                sample_air_images = np.array(sample_air_batch).astype(np.float32)
                sample_water_images = np.array(sample_water_batch).astype(np.float32)
                #print(sample_depth_batch.shape)
                #sample_depth_batch_f =np.array(sample_depth_batch).astype(np.float32)  
                #sample_depth_batch_n = np.multiply(self.max_depth,np.divide(sample_depth_batch_f,sample_depth_batch_f.max()))
                #sample_depth_batch_images = np.expand_dims(sample_depth_batch,axis=3)
                sample_depth_small_images = np.expand_dims(sample_depth_small_batch,axis=3)
                sample_depth_images = np.expand_dims(sample_depth_batch,axis=3)
                #sample_waterdepth_images = np.expand_dims(sample_waterdepth_batch,axis=3)
                print(sample_depth_images.shape)
                sample_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)

                samples = self.sess.run([self.wc_sampler],
                    feed_dict={self.sample_z:sample_z,self.sample_air_inputs:sample_air_images,self.sample_depth_inputs: sample_depth_images,self.depth_small_inputs:sample_depth_small_images})
#    feed_dict={self.z:sample_z,self.air_inputs:sample_air_images,self.depth_inputs: sample_depth_images})
                       # self.water_inputs: sample_water_images})

                #conv_ims = np.asarray(convs)
                #conv_ims = np.squeeze(conv_ims)
                sample_ims = np.asarray(samples)
                sample_ims = np.squeeze(sample_ims)
                sample_fake_images = sample_ims[:,0:self.sw,0:self.sh,0:3]
               # print("here")
               # print(sample_fake_images.shape)
                #sample_fake_images = scipy.misc.imresize(sample_fake_images,[64,64,64,3])
                #sample_fake_images=np.asarray(sample_fake_images)
               # print(sample_fake_images.shape)
                sample_fake_images_small = np.empty([0,self.sw,self.sh,3])
                for img_idx in range(0,self.batch_size):
                    out_name = "paper/fake_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                    sample_im = sample_ims[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_im = np.squeeze(sample_im)
                    #print(sample_im.shape)
                    scipy.misc.imsave(out_name,sample_im)
                    out_name = "paper/air_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                    sample_im = sample_air_images[img_idx,0:self.sw,0:self.sh,0:3]
                    #sample_im = conv_ims[img_idx,0:256,0:256,0:3]
                    sample_im = np.squeeze(sample_im)
                    scipy.misc.imsave(out_name,sample_im)
                    out_name = "paper/depth_%0d_%02d_%02d.mat" % (epoch, img_idx,idx)
                    sample_im = sample_depth_images[img_idx,0:self.sh,0:self.sw,0]
                    sample_im = np.squeeze(sample_im)
                    sio.savemat(out_name,{'depth':sample_im})
                    sample_fake = sample_fake_images[img_idx,0:self.sh,0:self.sw,0:3]
                    sample_fake = np.squeeze(sample_fake)
                    sample_fake = scipy.misc.imresize(sample_fake,[self.sw,self.sh,3],interp='bicubic')
                    sample_fake = np.expand_dims(sample_fake,axis=0)
                    sample_fake_images_small = np.append(sample_fake_images_small, sample_fake, axis=0)
                #print(sample_ims.shape)

                #print(sample_fake_images_small.shape)
                #disc_result = self.sess.run([self.sample_discriminator],
                #    feed_dict={self.sample_fake_inputs:sample_fake_images_small})
                #disc_result = np.asarray(disc_result)
                #disc_result = np.squeeze(disc_result)
                #disc_result = np.expand_dims(disc_result,axis=1)
                #disc_result1 = tf.nn.sigmoid_cross_entropy_with_logits(
                #    logits=disc_result, targets=tf.ones([64,1]))
                #disc_result2 = np.asarray(disc_result1)
                #disc_result[disc_result >= 0.5] = 1
                #disc_result[disc_result < 0.5] = 0
                #print(disc_result)
                #outfile = "my_results-mhl-c.txt"
                #fh = open(outfile,"a")
                #for img_idx in range(0,self.batch_size):
                #    out_name = "result_%0d_%02d_%02d.png" % (epoch, img_idx,idx)
                #    discwrite = "%d" % disc_result[img_idx]
                #    fh.write(out_name + " " + discwrite+"\n")
                #print(self.sess.run('logistic_loss_3:0'))
                #fh.close



        if np.mod(epoch, 5) == 1:
          self.save(config.checkpoint_dir, counter)
          print("saving checkpoint")

  def discriminator(self, image, depth=None,y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      if self.is_stereo:
        imd = tf.concat(3,[image,depth])
        h0 = lrelu(conv2d(imd, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      else:
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
      if self.is_stereo:
        imd = tf.concat(3,[image,depth])
        h0 = lrelu(conv2d(imd, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      else:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4)
      #return h4


  def wc_generator(self, z, image, depth, y=None):
    with tf.variable_scope("wc_generator") as scope:

      # Augment 1: water-based attenuation and backscatter
      with tf.variable_scope("g_atten"):
          init_r = tf.random_normal([1,1,1],mean=0.1,stddev=0.01,dtype=tf.float32)
          eta_r = tf.get_variable("g_eta_r",initializer=init_r)
          init_b = tf.random_normal([1,1,1],mean=0.1,stddev=0.01,dtype=tf.float32)
          eta_b = tf.get_variable("g_eta_b",initializer=init_b)
          init_g = tf.random_normal([1,1,1],mean=0.1,stddev=0.01,dtype=tf.float32)
          eta_g = tf.get_variable("g_eta_g",initializer=init_g)
          eta = tf.pack([eta_r,eta_g,eta_b],axis=3)
          init_b = tf.random_normal([1],mean=0.02,stddev=0.01,dtype=tf.float32)
          beta = tf.get_variable("g_beta",initializer=init_b)

          #init_br = tf.random_normal([1,1,1],mean=-0.01,stddev=0.01,dtype=tf.float32)
          #beta_r = tf.get_variable("g_beta_r",initializer=init_br)
          #init_bb = tf.random_normal([1,1,1],mean=-0.01,stddev=0.01,dtype=tf.float32)
          #beta_b = tf.get_variable("g_beta_b",initializer=init_bb)
          #init_bg = tf.random_normal([1,1,1],mean=-0.01,stddev=0.01,dtype=tf.float32)
          #beta_g = tf.get_variable("g_beta_g",initializer=init_bg)
          #beta = tf.pack([beta_r,beta_g,beta_b],axis=3)


          eta_d = tf.exp(tf.mul(-1.0,tf.mul(depth,eta)))
          B = tf.divide(beta,eta)
      h0 = tf.mul(image,eta_d)
#+tf.mul(B,(1.0-eta_d))

     ########################################################
      self.z_, self.h0z_w, self.h0z_b = linear(
          z, 64*64*64*1, 'g_h0_lin', with_w=True)

      self.h0z = tf.reshape(
          self.z_, [-1, 64, 64, 64*1])
      h0z = tf.nn.relu(self.g_bn0(self.h0z))
      h0z = tf.multiply(h0z,depth)

#      with tf.variable_scope('g_h1_conv'):
##          w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
#              initializer=tf.truncated_normal_initializer(stddev=0.02))
#      h1z = tf.nn.conv2d(h0z, w, strides=[1, 2, 2, 1], padding='SAME')
#      h1z = lrelu(self.g_bn1(h1z))

 #     with tf.variable_scope('g_h2_conv'):
 #         w2 = tf.get_variable('g_w_2',[ 5,5, h1z.get_shape()[-1], 1],
 #             initializer=tf.truncated_normal_initializer(stddev=0.02))
 #     h9z = tf.nn.conv2d(h1z, w2, strides=[1, 2, 2, 1], padding='SAME')
 #     h9z = lrelu(self.g_bn4(h9z))

  #    h8z, self.h2z_w, self.h2z_b = deconv2d(
  #        h9z, [64,32,32, 1], name='g_h2_deconv', with_w=True)


   #   self.h2z, self.h2z_w, self.h2z_b = deconv2d(
   #       h8z, [64,64,64, 1], name='g_h1_deconv', with_w=True)
      #h2z = tf.nn.relu(self.g_bn2(self.h2z))
     # h2z = tf.mul(tf.nn.sigmoid(self.h2z),255)
     ########################################################
    #  h2z = self.h2z
      #h_out = tf.mul(h2z, h0)


      with tf.variable_scope('g_h1_conv'):
          w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1z = tf.nn.conv2d(h0z, w, strides=[1, 1,1, 1], padding='SAME')
      h_g = lrelu(self.g_bn1(h1z))

      #self.h2z, self.h2z_w, self.h2z_b = deconv2d(
      #    h1z, [64,64,64,1], name='g_h1_deconv', with_w=True)
      #h_r = self.h2z

      with tf.variable_scope('g_h1_convr'):
          wr = tf.get_variable('g_wr',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 1, 1, 1], padding='SAME')
      h_r = lrelu(self.g_bn3(h1zr))

      #self.h2zr, self.h2z_wr, self.h2z_br = deconv2d(
      #    h1zr, [64,64,64, 1], name='g_h1_deconvr', with_w=True)
      #h_g = self.h2zr

      with tf.variable_scope('g_h1_convb'):
          wb = tf.get_variable('g_wb',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zb = tf.nn.conv2d(h0z, wb, strides=[1, 1, 1, 1], padding='SAME')
      h_b = lrelu(self.g_bn4(h1zb))

      #self.h2zb, self.h2z_wb, self.h2z_bb = deconv2d(
      #    h1zb, [64,64,64, 1], name='g_h1_deconvb', with_w=True)
      #h_b = self.h2zb

      #h2z = self.h2z
      h_r = tf.squeeze(h_r,axis=3)
      h_g = tf.squeeze(h_g,axis=3)
      h_b = tf.squeeze(h_b,axis=3)

      h_final=tf.pack([h_r,h_g,h_b],axis=3)




      h2 = tf.add(h_final,h0)

      # Augment 4: vignetting model
      kernlen = 64
      nsig = 2.2
      interval = (2*nsig+1.)/(kernlen)
      x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
      kern1d = np.diff(st.norm.cdf(x))
      kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
      kernel = (kernel_raw/kernel_raw.sum())/0.000495954880681
      kernel = np.expand_dims(kernel,axis=2)
      with tf.variable_scope("g_vig"):
          A = tf.get_variable('g_amp', [1],
              initializer=tf.truncated_normal_initializer(mean=0.7,stddev=0.3))
      h1 = tf.mul(A,kernel)
      #hav = tf.mul(h0,h1)
      h_out = tf.mul(h1,h2)
      return h_out, eta_r,eta_g,eta_b, beta

  def wc_sampler(self, z, image, depth, depth_small,y=None):
    with tf.variable_scope("wc_generator",reuse=True) as scope:
          # Augment 1: water-based attenuation and backscatter
      with tf.variable_scope("g_atten",reuse=True):
          init_r = tf.random_normal([1,1,1],mean=0.1,stddev=0.01,dtype=tf.float32)
          eta_r = tf.get_variable("g_eta_r",initializer=init_r)
          init_b = tf.random_normal([1,1,1],mean=0.1,stddev=0.01,dtype=tf.float32)
          eta_b = tf.get_variable("g_eta_b",initializer=init_b)
          init_g = tf.random_normal([1,1,1],mean=0.1,stddev=0.01,dtype=tf.float32)
          eta_g = tf.get_variable("g_eta_g",initializer=init_g)
          eta = tf.pack([eta_r,eta_g,eta_b],axis=3)
          init_b = tf.random_normal([1],mean=0.02,stddev=0.01,dtype=tf.float32)
          beta = tf.get_variable("g_beta",initializer=init_b)

         # init_br = tf.random_normal([1,1,1],mean=-0.01,stddev=0.01,dtype=tf.float32)
         # beta_r = tf.get_variable("g_beta_r",initializer=init_br)
         # init_bb = tf.random_normal([1,1,1],mean=-0.01,stddev=0.01,dtype=tf.float32)
         # beta_b = tf.get_variable("g_beta_b",initializer=init_bb)
         # init_bg = tf.random_normal([1,1,1],mean=-0.01,stddev=0.01,dtype=tf.float32)
         # beta_g = tf.get_variable("g_beta_g",initializer=init_bg)
         # beta = tf.pack([beta_r,beta_g,beta_b],axis=3)


          eta_d = tf.exp(tf.mul(-1.0,tf.mul(depth,eta)))
          B = tf.divide(beta,eta)
      h0 = tf.mul(image,eta_d)
#+tf.mul(B,(1.0-eta_d))


      self.z_, self.h0z_w, self.h0z_b = linear(
          z, 64*64*64*1, 'g_h0_lin', with_w=True)

      self.h0z = tf.reshape(
          self.z_, [-1, 64, 64, 64*1])
      h0z = tf.nn.relu(self.g_bn0(self.h0z))
      h0z = tf.multiply(h0z,depth_small)


    #  with tf.variable_scope('g_h1_conv',reuse=True):
    #      w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
    #          initializer=tf.truncated_normal_initializer(stddev=0.02))
    #  h1z = tf.nn.conv2d(h0z, w, strides=[1, 2, 2, 1], padding='SAME')
    #  h1z = lrelu(self.g_bn1(h1z))
    #  print(h1z)
    #  self.h2z, self.h2z_w, self.h2z_b = deconv2d(
    #      h1z, [64,64,64, 1], name='g_h1_deconv', with_w=True)
    #  print(self.h2z)
      #h2z = tf.nn.relu(self.g_bn2(self.h2z))
      #h2z = tf.mul(tf.nn.sigmoid(self.h2z),255)
     # h2z = self.h2z
     # h_out = tf.mul(h2z, h0)
      #print(h_out)


     # with tf.variable_scope('g1_conv',reuse=True):
     #     w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
     #         initializer=tf.truncated_normal_initializer(stddev=0.02))
     # h1z = tf.nn.conv2d(h0z, w, strides=[1, 2, 2, 1], padding='SAME')
     # h1z = lrelu(self.g_bn1(h1z))

      #self.h2z, self.h2z_w, self.h2z_b = deconv2d(
      #    h1z, [64,64,64,1], name='g_h1_deconv', with_w=True)
      #h_r = self.h2z

     # with tf.variable_scope('g_h1_convr',reuse=True):
     #     wr = tf.get_variable('g_wr',[ 5,5, h0z.get_shape()[-1], 1],
       #       initializer=tf.truncated_normal_initializer(stddev=0.02))
      #h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 2, 2, 1], padding='SAME')
     # h1zr = lrelu(self.g_bn3(h1zr))

      #self.h2zr, self.h2z_wr, self.h2z_br = deconv2d(
      #    h1zr, [64,64,64, 1], name='g_h1_deconvr', with_w=True)
      #h_g = self.h2zr

      #with tf.variable_scope('g_h1_convb',reuse=True):
      #    wb = tf.get_variable('g_wb',[ 5,5, h0z.get_shape()[-1], 1],
      #        initializer=tf.truncated_normal_initializer(stddev=0.02))
      #h1zb = tf.nn.conv2d(h0z, wb, strides=[1, 2, 2, 1], padding='SAME')
      #h1zb = lrelu(self.g_bn4(h1zb))

      #self.h2zb, self.h2z_wb, self.h2z_bb = deconv2d(
      #    h1zb, [64,64,64, 1], name='g_h1_deconvb', with_w=True)
      #h_b = self.h2zb

      #h2z = self.h2z
      #h_rxlt = tf.image.resize_images(h_r,[128,128],method=2)
      #h_gxlt = tf.image.resize_images(h_g,[128,128],method=2)
      #h_bxlt = tf.image.resize_images(h_b,[128,128],method=2)
      #h_rxl = tf.image.resize_images(h_rxlt,[256,256],method=2)
      #h_gxl = tf.image.resize_images(h_gxlt,[256,256],method=2)
      #h_bxl = tf.image.resize_images(h_bxlt,[256,256],method=2)



      with tf.variable_scope('g_h1_conv',reuse=True):
          w = tf.get_variable('g_w',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1z = tf.nn.conv2d(h0z, w, strides=[1, 1, 1, 1], padding='SAME')
      h_g = lrelu(self.g_bn1(h1z))

      #self.h2z, self.h2z_w, self.h2z_b = deconv2d(
      #    h1z, [64,64,64,1], name='g_h1_deconv', with_w=True)
      #h_r = self.h2z

      with tf.variable_scope('g_h1_convr',reuse=True):
          wr = tf.get_variable('g_wr',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 1, 1, 1], padding='SAME')
      h_r = lrelu(self.g_bn3(h1zr))

      #self.h2zr, self.h2z_wr, self.h2z_br = deconv2d(
      #    h1zr, [64,64,64, 1], name='g_h1_deconvr', with_w=True)
      #h_g = self.h2zr

      with tf.variable_scope('g_h1_convb',reuse=True):
          wb = tf.get_variable('g_wb',[ 5,5, h0z.get_shape()[-1], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
      h1zb = tf.nn.conv2d(h0z, wb, strides=[1,1,1, 1], padding='SAME')
      h_b = lrelu(self.g_bn4(h1zb))

      h_rxlt = tf.image.resize_images(h_r,[128,128],method=2)
      h_gxlt = tf.image.resize_images(h_g,[128,128],method=2)
      h_bxlt = tf.image.resize_images(h_b,[128,128],method=2)
      h_rxl = tf.image.resize_images(h_rxlt,[256,256],method=2)
      h_gxl = tf.image.resize_images(h_gxlt,[256,256],method=2)
      h_bxl = tf.image.resize_images(h_bxlt,[256,256],method=2)


      h_rxl = tf.squeeze(h_rxl,axis=3)
      h_gxl = tf.squeeze(h_gxl,axis=3)
      h_bxl = tf.squeeze(h_bxl,axis=3)

      print(h_r)

      print(h_rxl)
      h_final=tf.pack([h_rxl,h_gxl,h_bxl],axis=3)
      #h_finalxl = tf.image.resize_images(h_final,[256,256])
      h2 = tf.add(h_final,h0)


    #  h2 = tf.add(h_final,h0)

     # Augment 4: vignetting model
      kernlen = 64
      nsig = 2.2
      interval = (2*nsig+1.)/(kernlen)
      x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
      kern1d = np.diff(st.norm.cdf(x))
      kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
      kernel = (kernel_raw/kernel_raw.sum())/0.000495954880681
      kernel = np.expand_dims(kernel,axis=2)
      with tf.variable_scope("g_vig",reuse=True):
          A = tf.get_variable('g_amp', [1],
              initializer=tf.truncated_normal_initializer(mean=0.7,stddev=0.3))
      h1 = tf.mul(A,kernel)
      h1_xlt = tf.image.resize_images(h1,[128,128],method=2)
      h1_xl = tf.image.resize_images(h1_xlt,[256,256],method=2)


      #h5 = tf.expand_dims(h2z,axis=1)
      #h5 = tf.expand_dims(h5,axis=2)
      h_out = tf.mul(h1_xl,h2)
      #hav = tf.mul(h0,h1)
      return h2


      # Augment 1: water-based attenuation and backscatter
    #  with tf.variable_scope("g_atten",reuse=True):
        #  init_a = tf.random_uniform([1,1,1,3],minval=-0.6,maxval=0,dtype=tf.float32)
        #  eta = tf.get_variable("g_eta",initializer=init_a)
       #   init_b = tf.random_uniform([1],minval=0,maxval=0.01,dtype=tf.float32)
       #   beta = tf.get_variable("g_beta",initializer=init_b)
       #   eta_d = tf.exp(tf.mul(depth,eta))
      #h0 = tf.mul(image,eta_d)
      #print("here")
      #print(h0)
      #return h0
    #  # Augment 1: attenuation
    #  with tf.variable_scope("g_atten"):
    ##      init_a = tf.random_uniform([1,1,1,3],minval=-1,maxval=0,dtype=tf.float32)
    #      eta = tf.get_variable("g_eta",initializer=init_a)
    #      init_b = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
    ##      beta = tf.get_variable("g_beta",initializer=init_b)
     #     eta_d = tf.exp(tf.mul(depth,eta))
     # h0 = tf.mul(image,eta_d)+tf.mul(beta,(1.0-eta_d))


      # Augment 2: backscatter
#      with tf.variable_scope("g_bscatter"):#          init = tf.random_uniform([1],minval=0,maxval=256,dtype=tf.float32)
#          beta = tf.get_variable("g_beta",initializer=init)
#      h1 = h0 + tf.mul(beta,(1.0-tf.exp(eta_d)))

      #return h0

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
    depth=depth_mat["depth"]
    if self.is_crop:
      depth = scipy.misc.imresize(depth,(self.output_height,self.output_width),mode='F')
    depth = np.array(depth).astype(np.float32)
    depth = np.multiply(self.max_depth,np.divide(depth,depth.max()))

    return depth

  def read_depth_small(self, filename):
    depth_mat = sio.loadmat(filename)
    depth=depth_mat["depth"]
    if self.is_crop:
      depth = scipy.misc.imresize(depth,(64,64),mode='F')
    depth = np.array(depth).astype(np.float32)
    depth = np.multiply(self.max_depth,np.divide(depth,depth.max()))

    return depth

  def read_depth_sample(self, filename):
    depth_mat = sio.loadmat(filename)
    depth=depth_mat["depth"]
    if self.is_crop:
      depth = scipy.misc.imresize(depth,(self.sw,self.sh),mode='F')
    depth = np.array(depth).astype(np.float32)
    depth = np.multiply(self.max_depth,np.divide(depth,depth.max()))

    return depth

